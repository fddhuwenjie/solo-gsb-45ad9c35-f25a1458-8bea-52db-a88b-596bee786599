"""家庭应收台账的收款登记：唯一收款单号 + 正金额，按原入账顺序核销正额凭证。

规则（台账口径完全沿用 app/ledger.py 落盘的明细与凭证冻结计费栏，不重新核算）：
1. 调用方按账户提交**收款单号**（payment_no，全平台唯一）与**正金额**；
2. 核销口径在写事务内对账户当前台账重放得出：
   - **应退先抵**：已入账的应退（refund）凭证按入账顺序抵减「扣除已核销
     收款后仍待付」的正额凭证（full/supplement），抵不完的剩余退款形成
     **账户信用**（信用与尚欠互斥：有信用时必无未付正额凭证）；
   - **收款核销**：收款按**原入账顺序**（entry_no 升序）分配给抵减后仍
     欠付的正额凭证，**可部分付清**（最后一张只分到剩余额）；
   - **超付整拒**：收款金额超过当前应付（全部正额凭证尚欠合计）时
     **整笔拒绝，不留下任何收款流水**；
3. **逐笔冻结**：每笔收款的逐凭证分配、累计实收与收款后的尚欠/信用在
   落盘时冻结——后续再入账应补/应退凭证**不改写旧收款的分配**，只反映到
   查询接口按当前台账重放的账户级最新尚欠/信用上（允许显示信用）；
4. **幂等与冲突**：同一收款单号 + 同一账户 + 同一金额的重试返回**首次
   回执**（result=deduped，分配/累计/尚欠/信用全部保持首收时的冻结值，
   即使后来入账了新的应补/应退凭证）；单号换金额（同账户）或跨账户使用
   → 409 payment_no_conflict，不写入；
5. 收款核对与落盘和凭证入账**共用同一把 SQLite 写事务锁**（RLock +
   BEGIN IMMEDIATE）：并发收款在临界区内串行判定，payment_no 主键兜底，
   并发与进程重启后都不会重复核销；收款记录落在同一 SQLite（命名卷）。

账户级口径恒等式（重放不变量）：
    尚欠 = max(0, 台账净应收 − 累计实收)；信用 = max(0, 累计实收 − 台账净应收)

HTTP 错误码（整笔拒绝时均不写任何流水）：
- payment_exceeds_payable：收款金额超过当前应付（422，附 payable_yuan）；
- payment_no_conflict：收款单号已登记（换金额或跨账户使用）（409，
  附首次登记的 account_id / amount_yuan）；
- invalid_input：单号为空/超长、金额非正或非有限（422，请求模型拦截）。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.storage import SessionStore, get_store


class PaymentError(Exception):
    """收款操作失败：code 决定 HTTP 错误体 error 字段，status_code 决定状态码。"""

    def __init__(
        self,
        code: str,
        reason: str,
        status_code: int = 422,
        extra: Optional[dict] = None,
    ):
        super().__init__(reason)
        self.code = code
        self.status_code = status_code
        # 附加到错误响应体顶层的结构化字段（如当前应付、首次登记的账户/金额）
        self.extra = extra or {}


def _utc_now_iso() -> str:
    # 与 sessions/ledger._utc_now_iso 同口径：UTC、秒精度、带 Z。
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _r6(value: float) -> float:
    # 与台账入账同口径：金额统一保留 6 位小数（+0.0 消除 -0.0）。
    return round(float(value), 6) + 0.0


def _account_state(entry_rows, payment_rows):
    """重放账户台账与已落盘收款，得出每张正额凭证的尚欠、当前应付与信用。

    返回 (positives, payable, credit, total_received)：
    - positives：正额凭证（带符号金额 > 0，即 full/supplement）按入账次序
      排列，各带 amount_yuan（面额）、paid_yuan（已核销收款，冻结分配的
      累计）、remaining_yuan（面额 − 已核销）、outstanding_yuan（再扣除
      应退抵减后的尚欠）；
    - payable：当前应付 = 各正额凭证尚欠合计；
    - credit：账户信用 = 抵不完的剩余退款（与 payable 互斥）；
    - total_received：累计实收 = 全部已落盘收款金额之和。
    """
    paid_by_voucher = {}
    total_received = 0.0
    for row in payment_rows:
        total_received = _r6(total_received + float(row["amount_yuan"]))
        for alloc in json.loads(row["allocations_json"]):
            key = (alloc["session_id"], int(alloc["voucher_no"]))
            paid_by_voucher[key] = _r6(
                paid_by_voucher.get(key, 0.0) + float(alloc["allocated_yuan"])
            )

    positives: List[dict] = []
    refunds = 0.0
    for row in entry_rows:
        signed = float(row["signed_amount_yuan"])
        if signed > 0:
            key = (row["session_id"], int(row["voucher_no"]))
            paid = paid_by_voucher.get(key, 0.0)
            positives.append(
                {
                    "entry_no": int(row["entry_no"]),
                    "session_id": row["session_id"],
                    "voucher_no": int(row["voucher_no"]),
                    "amount_yuan": _r6(signed),
                    "paid_yuan": _r6(paid),
                    "remaining_yuan": _r6(signed - paid),
                    "outstanding_yuan": 0.0,  # 应退抵减后填写
                }
            )
        elif signed < 0:
            refunds = _r6(refunds + (-signed))

    # 应退先抵：按入账顺序抵减扣除已核销后仍待付的正额凭证，剩余形成信用。
    refund_left = refunds
    for pos in positives:
        offset = min(pos["remaining_yuan"], refund_left)
        pos["outstanding_yuan"] = _r6(pos["remaining_yuan"] - offset)
        refund_left = _r6(refund_left - offset)
    credit = refund_left
    payable = _r6(sum(pos["outstanding_yuan"] for pos in positives))
    return positives, payable, credit, total_received


def _payment_response(row, result: Optional[str]) -> dict:
    """把已落盘收款行组装成响应；分配与回执值全部取落盘时的冻结值。"""
    resp = {
        "payment_id": row["payment_id"],
        "payment_no": row["payment_no"],
        "account_id": row["account_id"],
        "payment_seq": int(row["payment_seq"]),
        "amount_yuan": float(row["amount_yuan"]),
        "allocations": json.loads(row["allocations_json"]),
        "cumulative_received_yuan": float(row["cumulative_received_yuan"]),
        "outstanding_after_yuan": float(row["outstanding_after_yuan"]),
        "credit_after_yuan": float(row["credit_after_yuan"]),
        "received_at": row["received_at"],
    }
    if result is not None:
        resp["result"] = result
    return resp


def receive_payment(
    account_id: str,
    payment_no: str,
    amount_yuan: float,
    store: Optional[SessionStore] = None,
) -> dict:
    """登记一笔到账款；超付整笔拒绝不留流水，同号同额重试返回首次回执。"""
    store = store or get_store()
    amount = _r6(amount_yuan)

    with store.write_transaction() as conn:
        # 1) 幂等 / 冲突判定优先于状态核对：同一单号永远对应首次的账户与金额，
        #    重试即使在当前状态下会超付，也原样返回首收回执。
        existing = store.get_payment_by_no(conn, payment_no)
        if existing is not None:
            first_account = existing["account_id"]
            first_amount = float(existing["amount_yuan"])
            if first_account != account_id:
                raise PaymentError(
                    "payment_no_conflict",
                    f"收款单号 {payment_no} 已在账户 {first_account} 登记收款 "
                    f"{first_amount} 元，同一单号不能再用于账户 {account_id}；"
                    "本次未登记任何收款",
                    409,
                    extra={
                        "payment_no": payment_no,
                        "account_id": first_account,
                        "amount_yuan": first_amount,
                    },
                )
            if abs(first_amount - amount) > 1e-9:
                raise PaymentError(
                    "payment_no_conflict",
                    f"收款单号 {payment_no} 在账户 {account_id} 已登记收款 "
                    f"{first_amount} 元，不能以金额 {amount} 元重复使用同一单号；"
                    "本次未登记任何收款",
                    409,
                    extra={
                        "payment_no": payment_no,
                        "account_id": first_account,
                        "amount_yuan": first_amount,
                    },
                )
            return _payment_response(existing, result="deduped")

        # 2) 重放台账与已落盘收款：应退先抵未付正额（剩余成信用），得出当前应付。
        positives, payable, credit, total_received = _account_state(
            store.list_ledger_entry_rows_in_txn(conn, account_id),
            store.list_payment_rows_in_txn(conn, account_id),
        )

        # 3) 超付整拒：超过当前应付的收款整笔拒绝，不留下任何流水。
        if amount > payable:
            raise PaymentError(
                "payment_exceeds_payable",
                f"收款金额 {amount} 元超过账户 {account_id} 当前应付 {payable} 元"
                f"（账户信用 {credit} 元）：整笔拒绝，不留下任何收款流水",
                422,
                extra={"payment_no": payment_no, "payable_yuan": payable},
            )

        # 4) 按原入账顺序把收款分配给尚未付清的正额凭证（可部分付清）。
        allocations: List[dict] = []
        left = amount
        for pos in positives:
            if left <= 0:
                break
            if pos["outstanding_yuan"] <= 0:
                continue
            take = _r6(min(pos["outstanding_yuan"], left))
            allocations.append(
                {
                    "entry_no": pos["entry_no"],
                    "session_id": pos["session_id"],
                    "voucher_no": pos["voucher_no"],
                    "allocated_yuan": take,
                }
            )
            left = _r6(left - take)

        # 5) 与上面全部核对在同一写事务内原子落盘；payment_no 主键兜底并发。
        payment_seq = store.count_payments(conn, account_id) + 1
        cumulative = _r6(total_received + amount)
        outstanding_after = _r6(payable - amount)
        store.insert_payment(
            conn,
            payment_id=str(uuid.uuid4()),
            payment_no=payment_no,
            account_id=account_id,
            payment_seq=payment_seq,
            amount_yuan=amount,
            allocations_json=json.dumps(allocations, ensure_ascii=False),
            cumulative_received_yuan=cumulative,
            outstanding_after_yuan=outstanding_after,
            credit_after_yuan=credit,
            received_at=_utc_now_iso(),
        )
        row = store.get_payment_by_no(conn, payment_no)
        return _payment_response(row, result="received")


def list_payments(
    account_id: str,
    store: Optional[SessionStore] = None,
) -> dict:
    """按收款次序查询账户全部收款（逐笔冻结），并给出随当前台账重放的
    累计实收、最新尚欠与信用（后续补退凭证改变最新口径，但不改写旧收款）。"""
    store = store or get_store()
    entry_rows = store.list_ledger_entry_rows(account_id)
    payment_rows = store.list_payment_rows(account_id)
    _, payable, credit, total_received = _account_state(entry_rows, payment_rows)
    return {
        "account_id": account_id,
        "payment_count": len(payment_rows),
        "total_received_yuan": total_received,
        "outstanding_yuan": payable,
        "credit_yuan": credit,
        "payments": [_payment_response(row, result=None) for row in payment_rows],
    }
