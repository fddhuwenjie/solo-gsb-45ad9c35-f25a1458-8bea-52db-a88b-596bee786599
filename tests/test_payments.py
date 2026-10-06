"""账户收款登记行为测试（应退抵减 → 信用 → FIFO 逐凭证核销）。

覆盖：
- 应退凭证按原入账顺序抵减尚未付清的正额凭证（与入账先后无关的全局应退池）、
  抵减剩余形成账户信用；收款按 entry_no 升序 FIFO 分配，可部分付清；
- 超过当前应付整笔拒绝且不留任何流水（收款主行与逐凭证核销都不写）；
- 无台账账户、非正金额拒绝；
- 同一单号 + 同账户 + 同金额重试返回首次冻结回执（result=deduped），即使后来
  入账新的应补/应退凭证，回执（分配、累计实收、当时尚欠/信用）不漂移；
- 同一单号换金额、跨账户使用 → 409 payment_no_conflict，附已登记账户/金额；
- 后续补退不改写旧收款分配：旧核销冻结，最新尚欠/信用随凭证重算，允许显示信用；
  信用随后续应补凭证被抵减；
- 逐凭证分配的 before/after 尚欠、累计实收、payment_count/payment_seq 次序；
- GET ledger 明细附逐凭证最新应收/抵减/已收/尚欠与账户级头寸；
- 收款记录列表按收款次序、内容为首收冻结值、账户级头寸为最新值；
- 并发：同一单号多线程提交只有一笔落盘（其余 deduped，或冲突），
  不同单号同账户并发串行、累计实收与尚欠一致、不重复核销；
- 按同一路径重建存储模拟进程重启：收款、核销、幂等重试仍正确；
- 旧库（无收款两表）打开自动补建；既有入账/出单等接口行为不变。
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.payments import submit_payment
from app.storage import configure_store

client = TestClient(app)


DISPATCH = dict(
    prices=[0.0, 1.0, 1.0, 0.0],
    base_load_kw=[0.2, 0.2, 0.2, 0.2],
    pv_kw=[0.0, 0.0, 0.0, 0.0],
    capacity_kwh=0.4,
    max_charge_kw=1.0,
    max_discharge_kw=1.0,
    charge_efficiency=0.9,
    discharge_efficiency=0.9,
    initial_soc_kwh=0.0,
    min_terminal_soc_kwh=0.0,
    degradation_cost_per_kwh=0.2,
)

RECORDS = [
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=round(0.2 / 0.9, 6), discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=round(0.1 + 0.2 / 0.9, 6),
        soc_start_kwh=0.0, soc_end_kwh=0.2,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.2, soc_end_kwh=0.1,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.1, soc_end_kwh=0.0,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=0.1,
        soc_start_kwh=0.0, soc_end_kwh=0.0,
    ),
]

TOL = 0.01


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    configure_store(str(tmp_path / "payments.db"))
    yield
    configure_store(None)


def _plan():
    return client.post("/dispatch", json=DISPATCH).json()


def _create_complete():
    sid = client.post(
        "/dispatch/sessions",
        json={"dispatch": DISPATCH, "plan_snapshot": _plan(), "audit_tolerance_kwh": TOL},
    ).json()["session_id"]
    assert client.post(
        f"/dispatch/sessions/{sid}/batches",
        json={"start_index": 0, "records": RECORDS[:2]},
    ).status_code == 200
    assert client.post(
        f"/dispatch/sessions/{sid}/batches",
        json={"start_index": 2, "records": RECORDS[2:]},
    ).status_code == 200
    return sid


def _rev(sid, expected, changes):
    return client.post(
        f"/dispatch/sessions/{sid}/revisions",
        json={"expected_revision": expected, "changes": changes},
    )


def _issue(sid, revision):
    return client.post(
        f"/dispatch/sessions/{sid}/vouchers",
        json={"expected_revision": revision},
    )


def _ch(index, record):
    return {"index": index, "record": record}


def _post_vouchers(account, items):
    return client.post(
        f"/accounts/{account}/ledger",
        json={"items": [{"session_id": s, "voucher_no": n} for s, n in items]},
    )


def _pay(account, payment_no, amount):
    return client.post(
        f"/accounts/{account}/payments",
        json={"payment_no": payment_no, "amount_yuan": amount},
    )


def _ledger(account):
    return client.get(f"/accounts/{account}/ledger").json()


def _payments(account):
    return client.get(f"/accounts/{account}/payments").json()


def _make_voucher_chain(sid):
    """凭证 1=full 0.10、2=supplement 0.01、3=refund 0.02（跳过 rev2 对 rev3 出单）。"""
    _issue(sid, 0)
    up1 = dict(RECORDS[1]); up1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    _rev(sid, 0, [_ch(1, up1)])
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "supplement"
    up2 = dict(RECORDS[1]); up2.update(base_load_kw=0.24, grid_purchase_energy_kwh=0.03)
    _rev(sid, 1, [_ch(1, up2)])
    down = dict(RECORDS[1]); down.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    _rev(sid, 2, [_ch(1, down)])
    v3 = _issue(sid, 3).json()
    assert v3["billing"]["type"] == "refund"
    assert v3["billing"]["amount_yuan"] == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# 基本核销：应退抵减、FIFO 分配、部分付清、信用
# ---------------------------------------------------------------------------


def test_refund_offsets_unpaid_positive_before_payment():
    """full 0.10 + supplement 0.01 − refund 0.02 已入账：应退先抵 full，
    收款时当前应付 = 0.09（full 剩 0.08、supplement 0.01），无信用。"""
    sid = _create_complete()
    _make_voucher_chain(sid)
    assert _post_vouchers("fam", [(sid, 1), (sid, 2), (sid, 3)]).status_code == 200

    led = _ledger("fam")
    assert led["outstanding_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert led["credit_yuan"] == 0.0
    assert led["total_received_yuan"] == 0.0
    # entry1 full 0.10 被应退抵掉 0.02，尚欠 0.08；entry2 supplement 0.01 未抵
    e1, e2, e3 = led["entries"]
    assert e1["refund_applied_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert e1["outstanding_yuan"] == pytest.approx(0.08, abs=1e-9)
    assert e2["refund_applied_yuan"] == 0.0
    assert e2["outstanding_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert e3["receivable_yuan"] == 0.0 and e3["outstanding_yuan"] == 0.0


def test_payment_fifo_partial_and_full_allocation():
    sid = _create_complete()
    _make_voucher_chain(sid)
    _post_vouchers("fam", [(sid, 1), (sid, 2), (sid, 3)])

    # 先收 0.05：全部分给 entry1（抵退后尚欠 0.08），entry1 仍欠 0.03
    r1 = _pay("fam", "RCPT-1", 0.05)
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["result"] == "received"
    assert b1["payment_count"] == 1
    assert b1["total_received_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert b1["credit_yuan"] == 0.0
    assert b1["outstanding_yuan"] == pytest.approx(0.04, abs=1e-9)
    alloc = b1["allocations"]
    assert len(alloc) == 1
    assert alloc[0]["allocation_no"] == 1
    assert alloc[0]["entry_no"] == 1 and alloc[0]["voucher_no"] == 1
    assert alloc[0]["allocated_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert alloc[0]["outstanding_before_yuan"] == pytest.approx(0.08, abs=1e-9)
    assert alloc[0]["outstanding_after_yuan"] == pytest.approx(0.03, abs=1e-9)
    assert b1["received_at"].endswith("Z")

    # 再收 0.04：0.03 付清 entry1，0.01 付清 entry2
    r2 = _pay("fam", "RCPT-2", 0.04)
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["payment_count"] == 2
    assert b2["total_received_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert b2["outstanding_yuan"] == 0.0
    assert [(a["entry_no"], a["allocated_yuan"]) for a in b2["allocations"]] == [
        (1, pytest.approx(0.03, abs=1e-9)),
        (2, pytest.approx(0.01, abs=1e-9)),
    ]
    # 跨凭证两张的 before/after 快照
    assert b2["allocations"][0]["outstanding_before_yuan"] == pytest.approx(0.03, abs=1e-9)
    assert b2["allocations"][0]["outstanding_after_yuan"] == 0.0
    assert b2["allocations"][1]["outstanding_before_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert b2["allocations"][1]["outstanding_after_yuan"] == 0.0

    led = _ledger("fam")
    assert led["outstanding_yuan"] == 0.0
    assert led["total_received_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert led["entries"][0]["paid_yuan"] == pytest.approx(0.08, abs=1e-9)
    assert led["entries"][1]["paid_yuan"] == pytest.approx(0.01, abs=1e-9)


def test_excess_payment_whole_rejected_without_any_trace():
    sid = _create_complete()
    _make_voucher_chain(sid)
    _post_vouchers("fam", [(sid, 1), (sid, 2), (sid, 3)])

    r = _pay("fam", "TOO-BIG", 0.10)  # 应付仅 0.09
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "payment_exceeds_outstanding"
    assert body["payment_no"] == "TOO-BIG"
    assert body["amount_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert body["outstanding_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert body["credit_yuan"] == 0.0

    # 整笔拒绝不留流水：收款列表为空、累计实收 0、逐凭证 paid 仍为 0
    assert _payments("fam")["payment_count"] == 0
    led = _ledger("fam")
    assert led["total_received_yuan"] == 0.0
    assert all(e["paid_yuan"] == 0.0 for e in led["entries"])

    # 恰好等于应付可以（边界）
    ok = _pay("fam", "EXACT", 0.09)
    assert ok.status_code == 200, ok.text
    assert ok.json()["outstanding_yuan"] == 0.0


def test_payment_equal_to_outstanding_boundary_accepted():
    sid = _create_complete()
    _issue(sid, 0)  # 仅 full 0.10
    _post_vouchers("a", [(sid, 1)])
    r = _pay("a", "EQ", 0.10)
    assert r.status_code == 200
    assert r.json()["outstanding_yuan"] == 0.0
    assert r.json()["allocations"][0]["outstanding_after_yuan"] == 0.0


# ---------------------------------------------------------------------------
# 信用：应退超过应收、只有信用时拒绝收款、信用抵后续应补
# ---------------------------------------------------------------------------


def test_credit_when_refund_exceeds_receivable():
    """full 0.10 与另一会话 full 0.10 已入账；一张应退 0.25（总费用不会为负，
    此处用直接存储构造账户级超额应退，验证信用口径与不设非负限制）。"""
    import uuid

    from app.storage import get_store

    s1 = _create_complete(); _issue(s1, 0)
    s2 = _create_complete(); _issue(s2, 0)
    _post_vouchers("cr", [(s1, 1), (s2, 1)])
    assert _ledger("cr")["outstanding_yuan"] == pytest.approx(0.20, abs=1e-9)

    # 直接补一条 0.25 的应退明细（绕过费用非负的业务约束，只测头寸口径）
    store = get_store()
    with store.write_transaction() as conn:
        conn.execute(
            "INSERT INTO account_ledger_entries(entry_id, account_id, entry_no, session_id,"
            " voucher_no, voucher_id, revision_no, billing_type, voucher_amount_yuan,"
            " relative_to_revision, relative_to_voucher_no, signed_amount_yuan,"
            " balance_after_yuan, total_cost_yuan, posted_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), "cr", 3, s1, 99, str(uuid.uuid4()), 0,
             "refund", 0.25, 0, 1, -0.25, -0.05, 0.0, "2026-01-01T00:00:00Z"),
        )

    led = _ledger("cr")
    # 0.25 应退抵掉两张 full 各 0.10，剩 0.05 信用；无尚欠
    assert led["outstanding_yuan"] == 0.0
    assert led["credit_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert led["balance_yuan"] == pytest.approx(-0.05, abs=1e-9)
    assert [e["refund_applied_yuan"] for e in led["entries"][:2]] == [
        pytest.approx(0.10, abs=1e-9), pytest.approx(0.10, abs=1e-9)
    ]

    # 只有信用、没有尚欠：任何正金额收款都整笔拒绝（附当前尚欠 0 与信用 0.05）
    r = _pay("cr", "NEG", 0.01)
    assert r.status_code == 422
    assert r.json()["error"] == "payment_exceeds_outstanding"
    assert r.json()["outstanding_yuan"] == 0.0
    assert r.json()["credit_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert _payments("cr")["payment_count"] == 0


def test_credit_consumed_by_later_supplement():
    """付清后入账的补退：未核销正额已为 0，应退全部形成信用；之后入账的应补
    凭证先由信用抵减（旧收款分配不被改写）。头寸口径用存储直造行精确控制。"""
    import uuid

    from app.payments import account_position
    from app.storage import get_store

    def _raw_entry(entry_no, sid, vno, signed, billing):
        store_local = get_store()
        with store_local.write_transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, dispatch_json,"
                " plan_snapshot_json, audit_tolerance_kwh, num_periods, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (sid, "{}", "{}", TOL, 4, "t"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO session_settlement_vouchers(voucher_id, session_id,"
                " voucher_no, revision_no, current_revision_at_issue, ledger_json,"
                " settlement_json, total_cost_yuan, billing_type, amount_yuan,"
                " relative_to_revision, relative_to_voucher_no, issued_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), sid, vno, 0, 0, "[]", "{}",
                 abs(signed), billing, abs(signed), None, None, "t"),
            )
            conn.execute(
                "INSERT INTO account_ledger_entries(entry_id, account_id, entry_no,"
                " session_id, voucher_no, voucher_id, revision_no, billing_type,"
                " voucher_amount_yuan, relative_to_revision, relative_to_voucher_no,"
                " signed_amount_yuan, balance_after_yuan, total_cost_yuan, posted_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), "fam", entry_no, sid, vno, str(uuid.uuid4()), 0,
                 billing, abs(signed), None, None, signed, signed, abs(signed), "t"),
            )

    # entry1 full 0.10 已被一笔收款全额付清
    _raw_entry(1, "cs1", 1, 0.10, "full")
    r = _pay("fam", "P1", 0.10)
    assert r.status_code == 200
    # entry2 refund 0.02（付清之后才入账）→ 全部信用
    _raw_entry(2, "cs2", 1, -0.02, "refund")
    led = _ledger("fam")
    assert led["outstanding_yuan"] == 0.0
    assert led["credit_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert led["entries"][0]["paid_yuan"] == pytest.approx(0.10, abs=1e-9)

    # entry3 supplement 0.01：信用先抵 0.01 → 仍无尚欠、信用剩 0.01
    _raw_entry(3, "cs1", 2, 0.01, "supplement")
    store = get_store()
    with store.write_transaction() as conn:
        pos = account_position(conn, store, "fam")
    assert pos["totals"]["outstanding_yuan"] == 0.0
    assert pos["totals"]["credit_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert pos["entries"][2]["refund_applied_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert pos["entries"][2]["outstanding_yuan"] == 0.0

    # entry4 supplement 0.02：信用再抵 0.01，尚欠 0.01、信用 0；旧收款不变
    _raw_entry(4, "cs1", 3, 0.02, "supplement")
    led = _ledger("fam")
    assert led["credit_yuan"] == 0.0
    assert led["outstanding_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert led["entries"][3]["refund_applied_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert led["entries"][3]["outstanding_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert led["entries"][0]["paid_yuan"] == pytest.approx(0.10, abs=1e-9)

    # 新收款只按最新尚欠收 0.01，全部给 entry4
    r2 = _pay("fam", "P2", 0.01)
    assert r2.status_code == 200, r2.text
    alloc = r2.json()["allocations"]
    assert [(a["entry_no"], a["allocated_yuan"]) for a in alloc] == [(4, pytest.approx(0.01, abs=1e-9))]
    assert r2.json()["outstanding_yuan"] == 0.0
    assert r2.json()["credit_yuan"] == 0.0

    # 旧回执 P1 冻结：当时累计实收 0.10、当时尚欠 0
    old = [p for p in _payments("fam")["payments"] if p["payment_no"] == "P1"][0]
    assert old["total_received_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert old["outstanding_yuan"] == 0.0


# ---------------------------------------------------------------------------
# 后续补退不改写旧收款分配；旧回执冻结
# ---------------------------------------------------------------------------


def test_later_refund_does_not_rewrite_old_payment_but_changes_latest_position():
    sid = _create_complete()
    _issue(sid, 0)  # full 0.10
    _post_vouchers("fam", [(sid, 1)])
    # 部分收款 0.06：entry1 尚欠 0.04
    r = _pay("fam", "OLD", 0.06)
    assert r.status_code == 200
    assert r.json()["outstanding_yuan"] == pytest.approx(0.04, abs=1e-9)

    # 之后才入账应退 0.02（凭证 2 应补 0.01、凭证 3 应退 0.02）
    up = dict(RECORDS[1]); up.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    _rev(sid, 0, [_ch(1, up)])
    _issue(sid, 1)  # supplement 0.01
    up2 = dict(RECORDS[1]); up2.update(base_load_kw=0.24, grid_purchase_energy_kwh=0.03)
    _rev(sid, 1, [_ch(1, up2)])
    down = dict(RECORDS[1]); down.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    _rev(sid, 2, [_ch(1, down)])
    _issue(sid, 3)  # refund 0.02
    _post_vouchers("fam", [(sid, 2), (sid, 3)])

    led = _ledger("fam")
    # 应退池 0.02：先抵 entry1 未核销部分 0.04 中的 0.02，再抵 entry2 0.01 不足
    # → entry1 尚欠 0.02、entry2 尚欠 0.01；旧现金分配仍 0.06 不变
    e1 = led["entries"][0]
    assert e1["paid_yuan"] == pytest.approx(0.06, abs=1e-9)  # 不被改写
    assert e1["refund_applied_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert e1["outstanding_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert led["outstanding_yuan"] == pytest.approx(0.03, abs=1e-9)
    assert led["credit_yuan"] == 0.0

    # 旧收款记录里冻结的逐凭证分配与当时快照不变
    old = [p for p in _payments("fam")["payments"] if p["payment_no"] == "OLD"][0]
    assert old["amount_yuan"] == pytest.approx(0.06, abs=1e-9)
    assert old["outstanding_yuan"] == pytest.approx(0.04, abs=1e-9)
    assert old["total_received_yuan"] == pytest.approx(0.06, abs=1e-9)
    assert len(old["allocations"]) == 1
    assert old["allocations"][0]["allocated_yuan"] == pytest.approx(0.06, abs=1e-9)
    assert old["allocations"][0]["outstanding_after_yuan"] == pytest.approx(0.04, abs=1e-9)


# ---------------------------------------------------------------------------
# 幂等重试与单号冲突
# ---------------------------------------------------------------------------


def test_same_payment_no_retry_returns_first_frozen_receipt():
    sid = _create_complete()
    _make_voucher_chain(sid)
    _post_vouchers("fam", [(sid, 1), (sid, 2), (sid, 3)])

    first = _pay("fam", "DUP", 0.05)
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["result"] == "received"

    # 之后又入账一张新应补凭证（改变当前应付），重试仍返回首次回执
    n1 = dict(RECORDS[1]); n1.update(base_load_kw=0.30, grid_purchase_energy_kwh=0.06)
    n2 = dict(RECORDS[2]); n2.update(base_load_kw=0.30, grid_purchase_energy_kwh=0.06)
    _rev(sid, 3, [_ch(1, n1), _ch(2, n2)])
    v4 = _issue(sid, 4).json()  # supplement 0.11
    assert v4["billing"]["type"] == "supplement"
    _post_vouchers("fam", [(sid, 4)])
    assert _ledger("fam")["outstanding_yuan"] != pytest.approx(0.04, abs=1e-9)

    retry = _pay("fam", "DUP", 0.05)
    assert retry.status_code == 200
    rb = retry.json()
    assert rb["result"] == "deduped"
    # 冻结字段全部等于首收（payment_count 也是当时的 1）
    for key in ("account_id", "payment_no", "amount_yuan", "received_at",
                "total_received_yuan", "outstanding_yuan", "credit_yuan", "payment_count"):
        assert rb[key] == first_body[key]
    assert rb["allocations"] == first_body["allocations"]
    # 不新增收款
    assert _payments("fam")["payment_count"] == 1


def test_same_no_different_amount_conflicts():
    sid = _create_complete()
    _issue(sid, 0)
    _post_vouchers("fam", [(sid, 1)])
    assert _pay("fam", "XNO", 0.05).status_code == 200

    r = _pay("fam", "XNO", 0.06)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "payment_no_conflict"
    assert body["payment_no"] == "XNO"
    assert body["account_id"] == "fam"
    assert body["amount_yuan"] == pytest.approx(0.05, abs=1e-9)
    # 冲突不写入：仍只有一笔
    assert _payments("fam")["payment_count"] == 1


def test_same_no_cross_account_conflicts():
    s1 = _create_complete(); _issue(s1, 0)
    s2 = _create_complete(); _issue(s2, 0)
    _post_vouchers("left", [(s1, 1)])
    _post_vouchers("right", [(s2, 1)])
    assert _pay("left", "SHARED", 0.05).status_code == 200

    r = _pay("right", "SHARED", 0.05)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "payment_no_conflict"
    assert body["account_id"] == "left"
    assert body["amount_yuan"] == pytest.approx(0.05, abs=1e-9)
    # 右账户没有这笔收款
    assert _payments("right")["payment_count"] == 0


# ---------------------------------------------------------------------------
# 非法输入
# ---------------------------------------------------------------------------


def test_non_positive_amount_rejected_by_validation():
    for bad in (0.0, -1.0):
        r = client.post(
            "/accounts/fam/payments", json={"payment_no": "Z", "amount_yuan": bad}
        )
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_input"


def test_empty_or_long_payment_no_rejected():
    r = client.post("/accounts/fam/payments", json={"payment_no": "", "amount_yuan": 1.0})
    assert r.status_code == 422
    r = client.post(
        "/accounts/fam/payments", json={"payment_no": "x" * 129, "amount_yuan": 1.0}
    )
    assert r.status_code == 422


def test_sub_micro_amount_rounded_to_zero_rejected_without_trace():
    """亚微级金额（1e-12，6 位舍入后为 0）不得生成空分配收款：整笔拒绝不留流水。"""
    import uuid

    from app.storage import get_store

    store = get_store()
    with store.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, dispatch_json, plan_snapshot_json,"
            " audit_tolerance_kwh, num_periods, created_at) VALUES (?,?,?,?,?,?)",
            ("u1", "{}", "{}", TOL, 4, "t"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO session_settlement_vouchers(voucher_id, session_id,"
            " voucher_no, revision_no, current_revision_at_issue, ledger_json,"
            " settlement_json, total_cost_yuan, billing_type, amount_yuan,"
            " relative_to_revision, relative_to_voucher_no, issued_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), "u1", 1, 0, 0, "[]", "{}",
             0.09, "full", 0.09, None, None, "t"),
        )
        conn.execute(
            "INSERT INTO account_ledger_entries(entry_id, account_id, entry_no, session_id,"
            " voucher_no, voucher_id, revision_no, billing_type, voucher_amount_yuan,"
            " relative_to_revision, relative_to_voucher_no, signed_amount_yuan,"
            " balance_after_yuan, total_cost_yuan, posted_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), "micro", 1, "u1", 1, str(uuid.uuid4()), 0,
             "full", 0.09, None, None, 0.09, 0.09, 0.09, "t"),
        )
    # 模型层 gt=0 放行 1e-12，服务层归一后判非正
    r = _pay("micro", "TINY", 1e-12)
    assert r.status_code == 422
    assert r.json()["error"] == "payment_amount_not_positive"
    assert _payments("micro")["payment_count"] == 0

    # 6 位小数边界：0.03 × 3 恰好付清 0.09（舍入不产生尾差，末笔正常 200）
    for i in range(3):
        assert _pay("micro", f"T{i}", 0.03).status_code == 200
    assert _ledger("micro")["outstanding_yuan"] == 0.0


def test_payment_to_account_without_ledger_rejected():
    r = _pay("no-ledger", "N1", 1.0)
    assert r.status_code == 422
    assert r.json()["error"] == "payment_empty_account"
    assert _payments("no-ledger")["payment_count"] == 0


# ---------------------------------------------------------------------------
# 收款记录查询
# ---------------------------------------------------------------------------


def test_payment_list_order_and_frozen_content_vs_latest_totals():
    sid = _create_complete()
    _make_voucher_chain(sid)
    _post_vouchers("fam", [(sid, 1), (sid, 2), (sid, 3)])
    _pay("fam", "P1", 0.05)
    _pay("fam", "P2", 0.04)

    lst = _payments("fam")
    assert lst["account_id"] == "fam"
    assert lst["payment_count"] == 2
    assert [p["payment_no"] for p in lst["payments"]] == ["P1", "P2"]
    assert all(p["result"] == "received" for p in lst["payments"])
    assert lst["payments"][0]["payment_count"] == 1
    assert lst["payments"][1]["payment_count"] == 2
    # 列表账户级字段为当前最新值
    assert lst["total_received_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert lst["outstanding_yuan"] == 0.0
    assert lst["credit_yuan"] == 0.0
    # 每条的快照字段为当时值（total_received_yuan 递增）
    assert lst["payments"][0]["total_received_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert lst["payments"][1]["total_received_yuan"] == pytest.approx(0.09, abs=1e-9)

    # 空账户查询
    empty = _payments("never-used")
    assert empty == {
        "account_id": "never-used",
        "payment_count": 0,
        "total_received_yuan": 0.0,
        "outstanding_yuan": 0.0,
        "credit_yuan": 0.0,
        "payments": [],
    }


# ---------------------------------------------------------------------------
# 并发：同单号只落一笔；不同单号串行一致、不重复核销
# ---------------------------------------------------------------------------


def test_concurrent_same_payment_no_only_one_lands():
    sid = _create_complete()
    _issue(sid, 0)
    _post_vouchers("race", [(sid, 1)])
    outcomes = []

    def worker():
        try:
            res = submit_payment("race", "RACE-NO", 0.10)
            outcomes.append(res["result"])
        except Exception as exc:  # noqa: BLE001
            outcomes.append(getattr(exc, "code", "error"))

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    received = [o for o in outcomes if o == "received"]
    deduped = [o for o in outcomes if o == "deduped"]
    assert len(received) == 1
    assert len(deduped) == 11
    assert set(outcomes) <= {"received", "deduped"}
    lst = _payments("race")
    assert lst["payment_count"] == 1
    assert lst["total_received_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert _ledger("race")["outstanding_yuan"] == 0.0


def test_concurrent_distinct_payments_serialize_without_double_writeoff():
    """多笔不同单号收款并发打到同一账户：全部落盘，FIFO 合计恰好付清，
    任一凭证 paid 不超过其应收、累计实收等于各笔之和。"""
    s1 = _create_complete(); _issue(s1, 0)  # 0.10
    s2 = _create_complete(); _issue(s2, 0)  # 0.10
    _post_vouchers("fam", [(s1, 1), (s2, 1)])
    errors = []
    # 应付 0.20，拆成 8 笔 0.025
    amounts = [0.025] * 8

    def worker(idx):
        try:
            submit_payment("fam", f"C{idx}", amounts[idx])
        except Exception as exc:  # noqa: BLE001
            errors.append((idx, getattr(exc, "code", str(exc))))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    led = _ledger("fam")
    assert led["outstanding_yuan"] == 0.0
    assert led["total_received_yuan"] == pytest.approx(0.20, abs=1e-9)
    for e in led["entries"]:
        assert e["paid_yuan"] <= e["receivable_yuan"] + 1e-9
    lst = _payments("fam")
    assert lst["payment_count"] == 8
    # 收款序号 1..8 连续
    assert sorted(p["payment_count"] for p in lst["payments"]) == list(range(1, 9))

    # 全部付清后再来的并发收款都整笔拒绝（不留流水）
    extra = []
    def over_worker():
        try:
            submit_payment("fam", "OVER", 0.01)
            extra.append("ok")
        except Exception as exc:  # noqa: BLE001
            extra.append(getattr(exc, "code", "error"))
    over_worker()
    assert extra == ["payment_exceeds_outstanding"]
    assert _payments("fam")["payment_count"] == 8


# ---------------------------------------------------------------------------
# 重启持久化、旧库补建、既有接口不变
# ---------------------------------------------------------------------------


def test_payments_persist_across_restart_and_retry_idempotent(tmp_path):
    db_path = str(tmp_path / "persist-pay.db")
    configure_store(db_path)
    sid = _create_complete()
    _make_voucher_chain(sid)
    _post_vouchers("fam", [(sid, 1), (sid, 2), (sid, 3)])
    _pay("fam", "P1", 0.05)

    configure_store(db_path)  # 模拟进程重启
    led = _ledger("fam")
    assert led["total_received_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert led["outstanding_yuan"] == pytest.approx(0.04, abs=1e-9)
    assert led["entries"][0]["paid_yuan"] == pytest.approx(0.05, abs=1e-9)

    lst = _payments("fam")
    assert lst["payment_count"] == 1
    assert lst["payments"][0]["payment_no"] == "P1"

    # 重启后同单号重试仍 deduped 且回执冻结
    retry = _pay("fam", "P1", 0.05)
    assert retry.status_code == 200
    assert retry.json()["result"] == "deduped"
    assert retry.json()["total_received_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert _payments("fam")["payment_count"] == 1

    # 重启后可继续登记收款，序号接续
    cont = _pay("fam", "P2", 0.04)
    assert cont.status_code == 200
    assert cont.json()["payment_count"] == 2
    assert cont.json()["outstanding_yuan"] == 0.0

    # 重启后超额仍拒绝
    over = _pay("fam", "P3", 0.01)
    assert over.status_code == 422
    assert over.json()["error"] == "payment_exceeds_outstanding"


def test_legacy_db_payment_tables_auto_created(tmp_path):
    """旧库（只有 sessions / session_records）打开即补建收款两表：可查可登记。"""
    import json
    import sqlite3

    from app.storage import get_store

    db_path = str(tmp_path / "legacy-pay.db")
    plan = client.post("/dispatch", json=DISPATCH).json()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, dispatch_json TEXT NOT NULL,
            plan_snapshot_json TEXT NOT NULL, audit_tolerance_kwh REAL NOT NULL,
            num_periods INTEGER NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE session_records (
            session_id TEXT NOT NULL, idx INTEGER NOT NULL, record_json TEXT NOT NULL,
            stored_at TEXT NOT NULL, PRIMARY KEY (session_id, idx));
        """
    )
    sid = "legacy-pay-session"
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
        (sid, json.dumps(DISPATCH), json.dumps(plan), TOL, 4, "2026-01-01T00:00:00Z"),
    )
    for idx, rec in enumerate(RECORDS):
        conn.execute(
            "INSERT INTO session_records VALUES (?, ?, ?, ?)",
            (sid, idx, json.dumps(rec), "2026-01-01T00:00:01Z"),
        )
    conn.commit()
    conn.close()

    configure_store(db_path)
    # 收款表自动补建：空账户收款查询为 0
    assert _payments("new-acct")["payment_count"] == 0
    # 端到端：出单 → 入账 → 收款
    assert _issue(sid, 0).status_code == 200
    assert _post_vouchers("new-acct", [(sid, 1)]).status_code == 200
    pay = _pay("new-acct", "LEGACY-PAY", 0.10)
    assert pay.status_code == 200, pay.text
    assert pay.json()["outstanding_yuan"] == 0.0
    # 存储层表结构就位
    store = get_store()
    with store._connect() as c:
        names = {
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"account_payments", "account_payment_allocations"} <= names


def test_ledger_posting_and_voucher_flows_unchanged():
    """新增收款不影响原入账、出单、会话查询行为（端到端抽验）。"""
    s1 = _create_complete()
    _issue(s1, 0)
    r = _post_vouchers("fam", [(s1, 1)])
    assert r.status_code == 200
    # 重复入账仍 409
    dup = _post_vouchers("fam", [(s1, 1)])
    assert dup.status_code == 409
    # 凭证列表仍可查
    vouchers = client.get(f"/dispatch/sessions/{s1}/vouchers")
    assert vouchers.status_code == 200
    assert [v["voucher_no"] for v in vouchers.json()["vouchers"]] == [1]
    # 台账明细冻结字段仍为入账口径（新增头寸列不破坏原字段）
    led = _ledger("fam")
    e = led["entries"][0]
    assert e["billing"]["type"] == "full"
    assert e["amount_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert e["balance_after_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert e["posted_at"].endswith("Z")
