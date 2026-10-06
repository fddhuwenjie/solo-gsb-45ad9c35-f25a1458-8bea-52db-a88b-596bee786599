"""家庭应收台账收款登记测试（唯一收款单号 + 正金额，按原入账顺序核销正额凭证）。

覆盖：
- 应退先抵未付正额、剩余退款形成信用；收款按原入账顺序分配（可部分付清）；
- 超过当前应付整笔拒绝且不留流水；空账户/已付清后再收一律拒绝；
- 逐凭证分配、累计实收、收款后尚欠/信用的返回与冻结；
- 同单号同账户同金额重试返回首次回执（即使后来入账新的应补/应退凭证，
  甚至在当前状态下该金额已超付）；单号换金额 / 跨账户使用 → 409；
- 后续补退不改写旧收款分配，最新尚欠/信用随凭证变化（允许显示信用）；
- 多笔收款累计实收；跨会话多凭证按入账顺序核销；no_adjustment 不产生应付；
- 并发同号只有一份生效、并发异号不超付；重启后收款仍在且不重复核销；
- 非法输入（非正/非有限金额、空/超长单号、缺字段）；
- 旧库自动补建收款表；既有台账/出单/会话接口行为不变。
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.payments import receive_payment
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
ORIG_TOTAL = 0.1


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


def _post_ledger(account, items):
    return client.post(
        f"/accounts/{account}/ledger",
        json={"items": [{"session_id": s, "voucher_no": n} for s, n in items]},
    )


def _pay(account, payment_no, amount):
    return client.post(
        f"/accounts/{account}/payments",
        json={"payment_no": payment_no, "amount_yuan": amount},
    )


def _get_payments(account):
    return client.get(f"/accounts/{account}/payments")


def _make_three_voucher_session():
    """满期会话：v1 全额 0.10、v2 应补 0.01、v3 应退 0.02（净应收 0.09）。"""
    sid = _create_complete()
    assert _issue(sid, 0).status_code == 200
    up = dict(RECORDS[1]); up.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 0, [_ch(1, up)]).status_code == 200
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "supplement"
    down = dict(RECORDS[1]); down.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 1, [_ch(1, down)]).status_code == 200
    v3 = _issue(sid, 2).json()
    assert v3["billing"]["type"] == "refund"
    assert v3["billing"]["amount_yuan"] == pytest.approx(0.02, abs=1e-9)
    return sid


# ---------------------------------------------------------------------------
# 核销口径：应退先抵、按入账顺序分配、可部分付清
# ---------------------------------------------------------------------------


def test_payment_allocates_after_refund_offset_in_posting_order():
    """e1 全额 0.10、e2 应补 0.01、e3 应退 0.02：应退先抵 e1（0.10→0.08），
    收款 0.05 全部分给 e1（部分付清），收款后尚欠 0.04。"""
    sid = _make_three_voucher_session()
    assert _post_ledger("fam", [(sid, 1), (sid, 2), (sid, 3)]).status_code == 200

    r = _pay("fam", "PAY-1", 0.05)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "received"
    assert body["payment_seq"] == 1
    assert body["amount_yuan"] == pytest.approx(0.05, abs=1e-9)
    # 应退 0.02 先抵 e1 → e1 尚欠 0.08；收款 0.05 全部核销 e1
    assert body["allocations"] == [
        {"entry_no": 1, "session_id": sid, "voucher_no": 1, "allocated_yuan": 0.05}
    ]
    assert body["cumulative_received_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert body["outstanding_after_yuan"] == pytest.approx(0.04, abs=1e-9)
    assert body["credit_after_yuan"] == 0.0
    assert body["received_at"].endswith("Z")

    # 再收 0.04 付清：e1 余 0.03 + e2 0.01，尚欠归零
    r = _pay("fam", "PAY-2", 0.04)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["payment_seq"] == 2
    assert [a["allocated_yuan"] for a in body["allocations"]] == pytest.approx(
        [0.03, 0.01], abs=1e-9
    )
    assert [a["entry_no"] for a in body["allocations"]] == [1, 2]
    assert body["cumulative_received_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert body["outstanding_after_yuan"] == 0.0


def test_payment_spans_multiple_sessions_in_entry_order():
    """两个会话各一张全额 0.10：收款 0.15 先付清 e1，再部分付 e2 0.05。"""
    s1 = _create_complete()
    s2 = _create_complete()
    _issue(s1, 0)
    _issue(s2, 0)
    assert _post_ledger("multi", [(s1, 1), (s2, 1)]).status_code == 200

    r = _pay("multi", "PAY-X", 0.15)
    assert r.status_code == 200, r.text
    allocs = r.json()["allocations"]
    assert [(a["session_id"], a["voucher_no"], a["allocated_yuan"]) for a in allocs] == [
        (s1, 1, 0.10), (s2, 1, 0.05)
    ]
    assert r.json()["outstanding_after_yuan"] == pytest.approx(0.05, abs=1e-9)

    detail = _get_payments("multi").json()
    assert detail["outstanding_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert detail["total_received_yuan"] == pytest.approx(0.15, abs=1e-9)


def test_no_adjustment_voucher_creates_no_payable():
    """no_adjustment 记 0 元明细但不产生应付：收款只核销全额凭证。"""
    sid = _create_complete()
    _issue(sid, 0)
    n3 = dict(RECORDS[3]); n3.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.11)
    assert _rev(sid, 0, [_ch(3, n3)]).status_code == 200
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "no_adjustment"
    assert _post_ledger("zero", [(sid, 1), (sid, 2)]).status_code == 200

    r = _pay("zero", "PAY-Z", 0.10)
    assert r.status_code == 200, r.text
    assert r.json()["allocations"] == [
        {"entry_no": 1, "session_id": sid, "voucher_no": 1, "allocated_yuan": 0.10}
    ]
    assert r.json()["outstanding_after_yuan"] == 0.0
    # 已付清：再收即超付
    assert _pay("zero", "PAY-Z2", 0.01).status_code == 422


# ---------------------------------------------------------------------------
# 超付整拒：不留流水
# ---------------------------------------------------------------------------


def test_overpayment_rejected_without_trace():
    sid = _create_complete()
    _issue(sid, 0)
    assert _post_ledger("over", [(sid, 1)]).status_code == 200

    r = _pay("over", "PAY-OVER", 0.11)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "payment_exceeds_payable"
    assert body["payable_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert body["payment_no"] == "PAY-OVER"
    # 不留流水
    detail = _get_payments("over").json()
    assert detail["payment_count"] == 0
    assert detail["payments"] == []
    assert detail["total_received_yuan"] == 0.0
    assert detail["outstanding_yuan"] == pytest.approx(0.10, abs=1e-9)

    # 恰好等于应付可收，收清后再收 0.01 也整拒
    assert _pay("over", "PAY-OK", 0.10).status_code == 200
    r = _pay("over", "PAY-OVER2", 0.01)
    assert r.status_code == 422
    assert r.json()["error"] == "payment_exceeds_payable"
    assert r.json()["payable_yuan"] == 0.0
    assert _get_payments("over").json()["payment_count"] == 1


def test_payment_on_empty_account_rejected():
    """从未入账的账户应付为 0，任何正金额收款都整笔拒绝。"""
    r = _pay("ghost", "PAY-G", 0.01)
    assert r.status_code == 422
    assert r.json()["error"] == "payment_exceeds_payable"
    assert r.json()["payable_yuan"] == 0.0
    assert _get_payments("ghost").json()["payment_count"] == 0


# ---------------------------------------------------------------------------
# 信用：应退大于未付正额时形成账户信用；最新口径随凭证变化
# ---------------------------------------------------------------------------


def test_refund_after_full_payment_forms_credit():
    """全额付清后入账应退 → 旧收款分配不变，账户显示信用；再收被拒。"""
    sid = _create_complete()
    _issue(sid, 0)
    assert _post_ledger("cred", [(sid, 1)]).status_code == 200
    paid = _pay("cred", "PAY-C", 0.10).json()
    assert paid["outstanding_after_yuan"] == 0.0

    # 更正使总费用 0.10 → 0.09：v2 应退 0.01 入账
    down = dict(RECORDS[1]); down.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 0, [_ch(1, down)]).status_code == 200
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "refund"
    assert _post_ledger("cred", [(sid, 2)]).status_code == 200

    detail = _get_payments("cred").json()
    # 旧收款分配与回执值不变（冻结）
    assert detail["payment_count"] == 1
    rec = detail["payments"][0]
    assert rec["payment_id"] == paid["payment_id"]
    assert rec["allocations"] == paid["allocations"]
    assert rec["outstanding_after_yuan"] == 0.0
    assert rec["cumulative_received_yuan"] == pytest.approx(0.10, abs=1e-9)
    # 最新口径：净应收 0.09 < 实收 0.10 → 信用 0.01
    assert detail["total_received_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert detail["outstanding_yuan"] == 0.0
    assert detail["credit_yuan"] == pytest.approx(0.01, abs=1e-9)

    # 有信用即无应付：再收 0.01 整拒
    r = _pay("cred", "PAY-C2", 0.01)
    assert r.status_code == 422
    assert r.json()["error"] == "payment_exceeds_payable"

    # 再入账应补 0.005（总费用 0.095）：信用抵减新正额 → 信用 0.005，仍无应付
    mid = dict(RECORDS[1]); mid.update(base_load_kw=0.19, grid_purchase_energy_kwh=0.005)
    assert _rev(sid, 1, [_ch(1, mid)]).status_code == 200
    v3 = _issue(sid, 2).json()
    assert v3["billing"]["type"] == "supplement"
    assert v3["billing"]["amount_yuan"] == pytest.approx(0.005, abs=1e-9)
    assert _post_ledger("cred", [(sid, 3)]).status_code == 200
    detail = _get_payments("cred").json()
    assert detail["outstanding_yuan"] == 0.0
    assert detail["credit_yuan"] == pytest.approx(0.005, abs=1e-9)


def test_credit_consumes_later_supplement_then_payable():
    """信用先抵后续应补，抵完仍有欠费时才可再收。"""
    sid = _create_complete()
    _issue(sid, 0)
    assert _post_ledger("cyc", [(sid, 1)]).status_code == 200
    _pay("cyc", "PAY-1", 0.10)
    # 应退 0.02（总费用 0.08）→ 信用 0.02
    down = dict(RECORDS[1]); down.update(base_load_kw=0.16, grid_purchase_energy_kwh=0.0)
    # t1 期望购电 = 0.5*0.16 - 0.09 = -0.01 → 截断 0；t2 也调低使总费用 0.08
    down2 = dict(RECORDS[2]); down2.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 0, [_ch(1, down), _ch(2, down2)]).status_code == 200
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "refund"
    assert v2["billing"]["amount_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert _post_ledger("cyc", [(sid, 2)]).status_code == 200
    assert _get_payments("cyc").json()["credit_yuan"] == pytest.approx(0.02, abs=1e-9)

    # 应补 0.05（总费用 0.13）：信用 0.02 抵后应付 0.03
    up = dict(RECORDS[1]); up.update(base_load_kw=0.24, grid_purchase_energy_kwh=0.03)
    up2 = dict(RECORDS[2]); up2.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 1, [_ch(1, up), _ch(2, up2)]).status_code == 200
    v3 = _issue(sid, 2).json()
    assert v3["billing"]["type"] == "supplement"
    assert v3["billing"]["amount_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert _post_ledger("cyc", [(sid, 3)]).status_code == 200

    detail = _get_payments("cyc").json()
    assert detail["credit_yuan"] == 0.0
    assert detail["outstanding_yuan"] == pytest.approx(0.03, abs=1e-9)
    # 收 0.04 超付整拒；收 0.03 成功且全核销 e3
    assert _pay("cyc", "PAY-2", 0.04).status_code == 422
    r = _pay("cyc", "PAY-2", 0.03)
    assert r.status_code == 200, r.text
    assert [a["entry_no"] for a in r.json()["allocations"]] == [3]
    assert r.json()["outstanding_after_yuan"] == 0.0


# ---------------------------------------------------------------------------
# 幂等与冲突：同号同额重试返回首收回执；换金额 / 跨账户 409
# ---------------------------------------------------------------------------


def test_retry_returns_first_receipt_frozen_after_new_vouchers():
    """重试返回首次回执：之后入账应退使当前口径变化，回执仍是首收时冻结值。"""
    sid = _make_three_voucher_session()  # e1 +0.10、e2 +0.01、e3 −0.02 → 应付 0.09
    assert _post_ledger("fz", [(sid, 1), (sid, 2), (sid, 3)]).status_code == 200
    first = _pay("fz", "PAY-F", 0.05).json()
    assert first["outstanding_after_yuan"] == pytest.approx(0.04, abs=1e-9)

    # 后续再入账一张应退 0.01（总费用 0.08）：当前应付变为 0.03，
    # 0.05 若按新状态发起会超付——但同号同额重试必须返回首收回执
    down2 = dict(RECORDS[2]); down2.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 2, [_ch(2, down2)]).status_code == 200
    v4 = _issue(sid, 3).json()
    assert v4["billing"]["type"] == "refund"
    assert _post_ledger("fz", [(sid, 4)]).status_code == 200
    assert _get_payments("fz").json()["outstanding_yuan"] == pytest.approx(0.03, abs=1e-9)

    r = _pay("fz", "PAY-F", 0.05)
    assert r.status_code == 200, r.text
    again = r.json()
    assert again["result"] == "deduped"
    assert again["payment_id"] == first["payment_id"]
    assert again["received_at"] == first["received_at"]
    assert again["payment_seq"] == first["payment_seq"]
    assert again["allocations"] == first["allocations"]
    assert again["cumulative_received_yuan"] == first["cumulative_received_yuan"]
    assert again["outstanding_after_yuan"] == first["outstanding_after_yuan"]
    # 不重复核销：仍只有一笔收款，累计实收不变
    detail = _get_payments("fz").json()
    assert detail["payment_count"] == 1
    assert detail["total_received_yuan"] == pytest.approx(0.05, abs=1e-9)


def test_payment_no_conflict_on_different_amount_same_account():
    sid = _create_complete()
    _issue(sid, 0)
    _post_ledger("dup", [(sid, 1)])
    assert _pay("dup", "PAY-D", 0.05).status_code == 200

    r = _pay("dup", "PAY-D", 0.06)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "payment_no_conflict"
    assert body["account_id"] == "dup"
    assert body["amount_yuan"] == pytest.approx(0.05, abs=1e-9)
    # 不写入：仍只有首笔
    assert _get_payments("dup").json()["payment_count"] == 1


def test_payment_no_conflict_across_accounts():
    sid = _create_complete()
    _issue(sid, 0)
    _post_ledger("left", [(sid, 1)])
    s2 = _create_complete()
    _issue(s2, 0)
    _post_ledger("right", [(s2, 1)])
    assert _pay("left", "PAY-X", 0.05).status_code == 200

    # 跨账户使用同一单号（同额与异额都冲突）
    for amount in (0.05, 0.02):
        r = _pay("right", "PAY-X", amount)
        assert r.status_code == 409
        body = r.json()
        assert body["error"] == "payment_no_conflict"
        assert body["account_id"] == "left"
    assert _get_payments("right").json()["payment_count"] == 0
    # 右账户用新单号可正常收款
    assert _pay("right", "PAY-Y", 0.05).status_code == 200


def test_cumulative_received_across_multiple_payments():
    sids = []
    for _ in range(3):
        sid = _create_complete()
        _issue(sid, 0)
        sids.append(sid)
    _post_ledger("cum", [(s, 1) for s in sids])

    r1 = _pay("cum", "P1", 0.10).json()
    r2 = _pay("cum", "P2", 0.05).json()
    r3 = _pay("cum", "P3", 0.15).json()
    assert r1["cumulative_received_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert r2["cumulative_received_yuan"] == pytest.approx(0.15, abs=1e-9)
    assert r3["cumulative_received_yuan"] == pytest.approx(0.30, abs=1e-9)
    assert r3["outstanding_after_yuan"] == 0.0
    assert [r["payment_seq"] for r in (r1, r2, r3)] == [1, 2, 3]

    detail = _get_payments("cum").json()
    assert detail["payment_count"] == 3
    assert [p["payment_no"] for p in detail["payments"]] == ["P1", "P2", "P3"]
    assert detail["total_received_yuan"] == pytest.approx(0.30, abs=1e-9)
    assert detail["outstanding_yuan"] == 0.0
    assert detail["credit_yuan"] == 0.0


def test_empty_account_payments_query():
    body = _get_payments("never-used").json()
    assert body == {
        "account_id": "never-used",
        "payment_count": 0,
        "total_received_yuan": 0.0,
        "outstanding_yuan": 0.0,
        "credit_yuan": 0.0,
        "payments": [],
    }


# ---------------------------------------------------------------------------
# 非法输入
# ---------------------------------------------------------------------------


def test_invalid_amount_and_payment_no():
    sid = _create_complete()
    _issue(sid, 0)
    _post_ledger("inv", [(sid, 1)])

    for bad in (0.0, -0.01, "Infinity"):
        r = client.post(
            "/accounts/inv/payments",
            content=f'{{"payment_no": "PAY-BAD", "amount_yuan": {bad}}}'
            if isinstance(bad, str)
            else None,
            json=None if isinstance(bad, str) else {"payment_no": "PAY-BAD", "amount_yuan": bad},
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422, (bad, r.text)
        assert r.json()["error"] == "invalid_input"

    # 空单号 / 空白单号 / 缺字段
    for payload in (
        {"payment_no": "", "amount_yuan": 0.05},
        {"payment_no": "   ", "amount_yuan": 0.05},
        {"amount_yuan": 0.05},
        {"payment_no": "PAY-BAD"},
    ):
        r = client.post("/accounts/inv/payments", json=payload)
        assert r.status_code == 422, payload
        assert r.json()["error"] == "invalid_input"
    assert _get_payments("inv").json()["payment_count"] == 0


# ---------------------------------------------------------------------------
# 并发：不重复核销、不超付
# ---------------------------------------------------------------------------


def test_concurrent_same_payment_no_only_one_received():
    sid = _create_complete()
    _issue(sid, 0)
    _post_ledger("race", [(sid, 1)])
    outcomes = []

    def worker(tag):
        try:
            res = receive_payment("race", "PAY-RACE", 0.10)
            outcomes.append((tag, res["result"], res["payment_id"]))
        except Exception as exc:  # noqa: BLE001
            outcomes.append((tag, getattr(exc, "code", "error"), None))

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    received = [o for o in outcomes if o[1] == "received"]
    deduped = [o for o in outcomes if o[1] == "deduped"]
    assert len(received) == 1
    assert len(deduped) == 9
    # 全部指向同一笔冻结收款
    assert {o[2] for o in outcomes} == {received[0][2]}
    detail = _get_payments("race").json()
    assert detail["payment_count"] == 1
    assert detail["total_received_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert detail["outstanding_yuan"] == 0.0


def test_concurrent_distinct_payments_never_overpay():
    """应付 0.10，6 个并发各收 0.04：恰好 2 笔成功（0.08），其余超付整拒。"""
    sid = _create_complete()
    _issue(sid, 0)
    _post_ledger("conc", [(sid, 1)])
    outcomes = []

    def worker(tag):
        try:
            res = receive_payment("conc", f"PAY-{tag}", 0.04)
            outcomes.append((tag, "received", res["payment_seq"]))
        except Exception as exc:  # noqa: BLE001
            outcomes.append((tag, getattr(exc, "code", "error"), None))

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    received = [o for o in outcomes if o[1] == "received"]
    rejected = [o for o in outcomes if o[1] == "payment_exceeds_payable"]
    assert len(received) == 2
    assert len(rejected) == 4
    detail = _get_payments("conc").json()
    assert detail["payment_count"] == 2
    assert detail["total_received_yuan"] == pytest.approx(0.08, abs=1e-9)
    assert detail["outstanding_yuan"] == pytest.approx(0.02, abs=1e-9)
    # 收款次序连续不重号
    assert sorted(p["payment_seq"] for p in detail["payments"]) == [1, 2]
    # 尾款 0.02 可收清
    assert _pay("conc", "PAY-LAST", 0.02).status_code == 200
    assert _get_payments("conc").json()["outstanding_yuan"] == 0.0


# ---------------------------------------------------------------------------
# 持久化：重启后收款仍在、不重复核销；旧库自动补建收款表
# ---------------------------------------------------------------------------


def test_payments_persist_across_restart(tmp_path):
    db_path = str(tmp_path / "persist-payments.db")
    configure_store(db_path)
    sid = _make_three_voucher_session()
    _post_ledger("fam", [(sid, 1), (sid, 2), (sid, 3)])
    first = _pay("fam", "PAY-P", 0.05).json()

    configure_store(db_path)  # 模拟进程重启
    detail = _get_payments("fam").json()
    assert detail["payment_count"] == 1
    rec = detail["payments"][0]
    assert rec["payment_id"] == first["payment_id"]
    assert rec["allocations"] == first["allocations"]
    assert rec["cumulative_received_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert detail["outstanding_yuan"] == pytest.approx(0.04, abs=1e-9)

    # 重启后重试仍取回首收回执，不重复核销
    again = _pay("fam", "PAY-P", 0.05).json()
    assert again["result"] == "deduped"
    assert again["payment_id"] == first["payment_id"]
    assert _get_payments("fam").json()["total_received_yuan"] == pytest.approx(0.05, abs=1e-9)

    # 重启后新收款次序接续、核销基于持久化状态
    nxt = _pay("fam", "PAY-Q", 0.04).json()
    assert nxt["result"] == "received"
    assert nxt["payment_seq"] == 2
    assert nxt["outstanding_after_yuan"] == 0.0


def test_legacy_db_payments_table_auto_created(tmp_path):
    """旧库（只有 sessions / session_records）打开即补建收款表，可查可收。"""
    import json
    import sqlite3

    db_path = str(tmp_path / "legacy-payments.db")
    plan = client.post("/dispatch", json=DISPATCH).json()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            dispatch_json TEXT NOT NULL,
            plan_snapshot_json TEXT NOT NULL,
            audit_tolerance_kwh REAL NOT NULL,
            num_periods INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE session_records (
            session_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            stored_at TEXT NOT NULL,
            PRIMARY KEY (session_id, idx)
        );
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
    assert _get_payments("legacy-acct").json()["payment_count"] == 0
    # 旧库无凭证/台账表 → 出单、入账、收款端到端可用
    assert _issue(sid, 0).status_code == 200
    assert _post_ledger("legacy-acct", [(sid, 1)]).status_code == 200
    r = _pay("legacy-acct", "PAY-L", 0.10)
    assert r.status_code == 200, r.text
    assert r.json()["outstanding_after_yuan"] == 0.0


# ---------------------------------------------------------------------------
# 既有接口行为不变
# ---------------------------------------------------------------------------


def test_existing_endpoints_unaffected_by_payments():
    """收款不改台账余额口径：台账查询、出单重试、会话查询行为不变。"""
    sid = _make_three_voucher_session()
    _post_ledger("mix", [(sid, 1), (sid, 2), (sid, 3)])
    _pay("mix", "PAY-M", 0.05)

    ledger = client.get("/accounts/mix/ledger").json()
    assert ledger["entry_count"] == 3
    assert ledger["balance_yuan"] == pytest.approx(0.09, abs=1e-9)

    # 出单重试仍幂等返回原凭证
    again = _issue(sid, 0).json()
    assert again["result"] == "existing"
    # 会话查询正常
    sess = client.get(f"/dispatch/sessions/{sid}").json()
    assert sess["status"] == "complete"
    # 空账户台账查询口径不变
    assert client.get("/accounts/never-used/ledger").json() == {
        "account_id": "never-used",
        "entry_count": 0,
        "balance_yuan": 0.0,
        "entries": [],
    }
