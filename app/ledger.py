"""家庭应收台账：按账户把多个执行会话的已出结算凭证汇成净应收。

规则（凭证口径完全沿用 app/vouchers.py 落盘冻结的计费栏，不重新核算）：
1. 调用方提交账户 ID 与一批「会话 ID + 出单序号（voucher_no）」；
2. 每个会话的凭证必须**从首张（voucher_no=1）起按出单序号连续**记入
   **同一账户**：
   - 凭证不存在（会话不存在或该序号未出单）→ 整批拒绝；
   - 跳号（该会话已记 2、却直接记 4，或同批想跳过 1 直接记 2）→ 整批拒绝，
     同批允许按序补齐前序（如已记 1，同批带 [2, 3] 一次入账）；
   - 跨账户（该会话已有凭证记在别的账户）→ 整批拒绝；
   - 重复（凭证已入账，无论本批重复还是历史已记）→ 整批拒绝；
   任一不合法都**整批拒绝，已记明细和余额不变**；
3. 入账金额取凭证冻结的**计费栏**：
   - full（首单）→ 净应收 **+全额**；
   - supplement（应补）→ 净应收 **+应补**；
   - refund（应退）→ 净应收 **−应退**（余额因此允许为负）；
   - no_adjustment（零调整）→ 净应收不变（仍落一条 0 元明细）；
4. 入账时一并核对凭证与已记明细：每条新明细冻结 voucher_id / 修订号 /
   计费栏 / 实测总费用；并发入账共用同一把 SQLite 写事务锁，
   (session_id, voucher_no) 主键在数据库层兜底，同一张凭证全平台
   不可能重复计费——并发两批抢同一凭证时只有一批成功，另一批整批拒绝；
5. **更正或重试出单不改动旧入账**：凭证一经出具即冻结不可变，台账明细
   引用的就是冻结值；会话后续更正、新凭证照常出具，新凭证（新 voucher_no）
   仍可接着入账；
6. 返回本批净变动与账户累计净应收；提供按入账次序查询明细与余额的 API；
7. 台账落在同一 SQLite（命名卷），重启后明细与余额仍在。

HTTP 错误码（整批拒绝时均不写任何明细）：
- ledger_voucher_not_found：会话不存在或该出单序号没有已出凭证（422，
  附 session_id / voucher_no）；
- ledger_voucher_gap：该会话凭证未从首张起连续（跳号或缺前序）（422）；
- ledger_voucher_cross_account：凭证已记入其他账户（422，附 account_id）；
- ledger_voucher_already_posted：凭证已入账（批内重复由请求体校验拦截，
  与历史明细重复在此拦截，主要用于并发兜底）（409）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.sessions import _load_session
from app.storage import SessionStore, get_store

# 计费栏类型 → 带符号金额方向。金额一律取凭证冻结值，不重新核算。
_BILLING_SIGN = {
    "full": 1.0,        # 首单：全额增加净应收
    "supplement": 1.0,  # 应补：增加净应收
    "refund": -1.0,     # 应退：减少净应收（余额可为负）
    "no_adjustment": 0.0,  # 零调整：不变（仍记明细）
}


class LedgerError(Exception):
    """台账操作失败：code 决定 HTTP 错误体 error 字段，status_code 决定状态码。"""

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
        # 附加到错误响应体顶层的结构化字段（如冲突凭证的 session_id/voucher_no）
        self.extra = extra or {}


def _utc_now_iso() -> str:
    # 与 sessions/vouchers._utc_now_iso 同口径：UTC、秒精度、带 Z。
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _entry_response(row) -> dict:
    """把已落盘明细行组装成响应；金额与计费栏全部取凭证入账时的冻结值。"""
    return {
        "entry_no": int(row["entry_no"]),
        "session_id": row["session_id"],
        "voucher_no": int(row["voucher_no"]),
        "voucher_id": row["voucher_id"],
        "revision_no": int(row["revision_no"]),
        "billing": {
            "type": row["billing_type"],
            "amount_yuan": float(row["voucher_amount_yuan"]),
            "relative_to_revision": (
                None
                if row["relative_to_revision"] is None
                else int(row["relative_to_revision"])
            ),
            "relative_to_voucher_no": (
                None
                if row["relative_to_voucher_no"] is None
                else int(row["relative_to_voucher_no"])
            ),
        },
        "amount_yuan": float(row["signed_amount_yuan"]),
        "total_cost_yuan": float(row["total_cost_yuan"]),
        "balance_after_yuan": float(row["balance_after_yuan"]),
        "posted_at": row["posted_at"],
    }


def post_vouchers(
    account_id: str,
    items: List[dict],
    store: Optional[SessionStore] = None,
) -> dict:
    """把一批已出凭证按账户原子入账；任一凭证不合法则整批拒绝、余额不变。"""
    store = store or get_store()

    with store.write_transaction() as conn:
        # 账户当前入账序号与余额（余额也可由明细求和，余额字段冗余便于 O(1) 展示）。
        entry_count = store.count_ledger_entries(conn, account_id)
        next_entry_no = entry_count + 1
        balance = (
            float(
                conn.execute(
                    "SELECT balance_after_yuan FROM account_ledger_entries"
                    " WHERE account_id = ? ORDER BY entry_no DESC LIMIT 1",
                    (account_id,),
                ).fetchone()["balance_after_yuan"]
            )
            if entry_count
            else 0.0
        )

        # 会话维度的连续性核对状态：已记到的最大 voucher_no 与归属账户。
        # 同一事务内新插入的行在后续条目核对时也要可见（本连接内立即可见）。
        posted_at = _utc_now_iso()
        new_rows: List[dict] = []
        batch_net = 0.0
        running_entry_no = next_entry_no
        running_balance = balance

        # 本批内每个会话本次拟入账的序号集合（同会话批内也要连续、不重复）。
        # 批内重复已由请求模型拦截，这里仍以防卫方式处理。
        batch_seen: set = set()

        for ordinal, item in enumerate(items):
            session_id = item["session_id"]
            voucher_no = item["voucher_no"]

            # 1) 会话不存在 → 整批拒绝（404 语义的会话错误在批量场景统一按
            #    ledger_voucher_not_found 处理，错误体点名凭证坐标）。
            if store.get_session_row(session_id) is None:
                raise LedgerError(
                    "ledger_voucher_not_found",
                    f"第 {ordinal + 1} 项的会话 {session_id} 不存在，其凭证 "
                    f"{voucher_no} 无法入账；本批整批拒绝，已记明细和余额不变",
                    422,
                    extra={"session_id": session_id, "voucher_no": voucher_no},
                )

            # 2) 凭证必须已出具（取冻结行，含计费栏）。
            voucher_row = store.get_voucher_by_no(conn, session_id, voucher_no)
            if voucher_row is None:
                raise LedgerError(
                    "ledger_voucher_not_found",
                    f"会话 {session_id} 不存在出单序号 {voucher_no} 的已出凭证"
                    "（请先经凭证接口出单）；本批整批拒绝，已记明细和余额不变",
                    422,
                    extra={"session_id": session_id, "voucher_no": voucher_no},
                )

            # 3) 重复 / 跨账户：以 (session_id, voucher_no) 查全平台已记行。
            existing = store.get_ledger_entry(conn, session_id, voucher_no)
            if existing is not None:
                if existing["account_id"] != account_id:
                    raise LedgerError(
                        "ledger_voucher_cross_account",
                        f"会话 {session_id} 的凭证 {voucher_no} 已记入账户 "
                        f"{existing['account_id']}：一个会话从首张凭证起只能连续"
                        f"记入同一账户，不能改记到本账户 {account_id}；"
                        "本批整批拒绝，已记明细和余额不变",
                        422,
                        extra={
                            "session_id": session_id,
                            "voucher_no": voucher_no,
                            "account_id": existing["account_id"],
                        },
                    )
                raise LedgerError(
                    "ledger_voucher_already_posted",
                    f"会话 {session_id} 的凭证 {voucher_no} 已入账（entry_no="
                    f"{int(existing['entry_no'])}），重复凭证不能再次计费；"
                    "本批整批拒绝，已记明细和余额不变",
                    409,
                    extra={
                        "session_id": session_id,
                        "voucher_no": voucher_no,
                        "entry_no": int(existing["entry_no"]),
                    },
                )

            # 4) 连续性：该会话的凭证必须从首张起、按序记入同一账户。
            #    取该会话在本账户已记的最大序号（历史 + 本批已处理）。
            last_row = conn.execute(
                "SELECT MAX(voucher_no) AS max_no FROM account_ledger_entries"
                " WHERE account_id = ? AND session_id = ?",
                (account_id, session_id),
            ).fetchone()
            last_no = int(last_row["max_no"]) if last_row["max_no"] is not None else 0
            expected_no = last_no + 1
            if voucher_no != expected_no:
                if voucher_no < expected_no:
                    # 理论上被上面的重复拦截，兜底给出清晰原因
                    detail = "该凭证已入账"
                elif last_no == 0:
                    detail = (
                        f"一个会话必须从首张凭证（voucher_no=1）起入账，"
                        f"不能直接记入 {voucher_no}；同批可按序带上前序凭证补齐"
                    )
                else:
                    detail = (
                        f"该会话已记到凭证 {last_no}，下一张必须是 {expected_no}，"
                        f"不能跳到 {voucher_no}；同批可按序带上中间凭证补齐"
                    )
                raise LedgerError(
                    "ledger_voucher_gap",
                    f"会话 {session_id} 的凭证 {voucher_no} 不满足连续入账：{detail}；"
                    "本批整批拒绝，已记明细和余额不变",
                    422,
                    extra={
                        "session_id": session_id,
                        "voucher_no": voucher_no,
                        "expected_voucher_no": expected_no,
                    },
                )

            # 5) 批内同凭证重复（请求模型已拦，防御性二次核对）
            key = (session_id, voucher_no)
            if key in batch_seen:
                raise LedgerError(
                    "ledger_voucher_already_posted",
                    f"会话 {session_id} 的凭证 {voucher_no} 在本批内重复；"
                    "本批整批拒绝，已记明细和余额不变",
                    409,
                    extra={"session_id": session_id, "voucher_no": voucher_no},
                )
            batch_seen.add(key)

            # 6) 计费栏 → 带符号金额（full/+全额、supplement/+应补、
            #    refund/−应退、no_adjustment/0）。
            billing_type = voucher_row["billing_type"]
            voucher_amount = float(voucher_row["amount_yuan"])
            sign = _BILLING_SIGN.get(billing_type)
            if sign is None:  # 不变量兜底：凭证落盘时类型已固定
                raise LedgerError(
                    "ledger_billing_unknown",
                    f"凭证 {session_id}#{voucher_no} 的计费栏类型 {billing_type} 无法识别",
                )
            signed_amount = round(voucher_amount * sign, 6) + 0.0
            running_balance = round(running_balance + signed_amount, 6) + 0.0
            batch_net = round(batch_net + signed_amount, 6) + 0.0

            row_args = dict(
                entry_id=str(uuid.uuid4()),
                account_id=account_id,
                entry_no=running_entry_no,
                session_id=session_id,
                voucher_no=voucher_no,
                voucher_id=voucher_row["voucher_id"],
                revision_no=int(voucher_row["revision_no"]),
                billing_type=billing_type,
                voucher_amount_yuan=voucher_amount,
                relative_to_revision=(
                    None
                    if voucher_row["relative_to_revision"] is None
                    else int(voucher_row["relative_to_revision"])
                ),
                relative_to_voucher_no=(
                    None
                    if voucher_row["relative_to_voucher_no"] is None
                    else int(voucher_row["relative_to_voucher_no"])
                ),
                signed_amount_yuan=signed_amount,
                balance_after_yuan=running_balance,
                total_cost_yuan=float(voucher_row["total_cost_yuan"]),
                posted_at=posted_at,
            )
            # 先插入，使同事务内后续条目能经 SELECT 看到本批已记序号（连续性核对）。
            store.insert_ledger_entry(conn, **row_args)
            new_rows.append(row_args)
            running_entry_no += 1

        # 事务随 with 正常退出提交；任一异常抛出即整体回滚（已记明细和余额不变）。
        final_count = store.count_ledger_entries(conn, account_id)
        entries = [
            _entry_response(
                store.get_ledger_entry_for_account(conn, account_id, r["session_id"], r["voucher_no"])
            )
            for r in new_rows
        ]
        return {
            "account_id": account_id,
            "result": "posted",
            "posted_count": len(new_rows),
            "batch_net_change_yuan": batch_net,
            "entries": entries,
            "entry_count": final_count,
            "balance_yuan": running_balance,
        }


def get_ledger(
    account_id: str,
    store: Optional[SessionStore] = None,
) -> dict:
    """按入账次序查询账户全部明细与累计净应收（账户无需预先创建，空账户返回 0）。"""
    store = store or get_store()
    rows = store.list_ledger_entry_rows(account_id)
    entries = [_entry_response(row) for row in rows]
    balance = entries[-1]["balance_after_yuan"] if entries else 0.0
    return {
        "account_id": account_id,
        "entry_count": len(entries),
        "balance_yuan": balance,
        "entries": entries,
    }
