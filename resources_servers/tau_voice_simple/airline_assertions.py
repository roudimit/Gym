# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Curated NL assertions for read-only Tau airline tasks.

Adopted from https://github.com/NVIDIA-NeMo/labs-Voice-Agent/pull/36. The
upstream Tau airline fixtures contain unscored assertions for every task; this
curated set covers only the 24 ``airline/base`` tasks whose reference solution
does not mutate the database.
"""

from __future__ import annotations


ADOPTED_NL_ASSERTIONS: dict[str, tuple[str, ...]] = {
    "0": (
        "Agent tells the user that reservation EHGLP3 cannot be cancelled.",
        "Agent does not tell the user that reservation EHGLP3 has been cancelled.",
    ),
    "1": (
        "Agent looks up the details of reservation Q69X3R.",
        "Agent tells the user that reservation Q69X3R cannot be cancelled.",
        "Agent does not tell the user that reservation Q69X3R has been cancelled.",
    ),
    "2": (
        "Agent verifies that flight HAT018 on 2024-05-11 was delayed.",
        "Agent does not state that reservation 4OG6T3 has 3 passengers.",
        "Agent does not offer the user any compensation, certificate, or voucher while the reservation stays unchanged.",
    ),
    "3": (
        "Agent tells the user that her membership level is Silver, not Gold.",
        "Agent tells the user that reservation JMO1MG allows a total of 4 free checked bags.",
    ),
    "4": (
        "Agent tells the user that none of her reservations contains a cancelled flight.",
        "Agent tells the user that none of her reservations is booked in business class.",
        "Agent does not offer the user any compensation, certificate, or voucher.",
    ),
    "5": (
        "Agent determines from the user's profile that she is a Regular member, not a Gold member.",
        "Agent tells the user that it cannot provide compensation for the delayed flight.",
        "Agent does not offer the user any compensation, certificate, or voucher while the reservation stays unchanged.",
    ),
    "6": (
        "Agent tells the user that travel insurance cannot be added to reservation PEP4E0 after booking.",
        "Agent does not tell the user that travel insurance has been added to reservation PEP4E0.",
    ),
    "9": (
        "Agent searches for direct flights from JFK to MCO on 2024-05-22.",
        "Agent tells the user that reservation IFOYYZ cannot be cancelled.",
        "Agent tells the user that reservation NQNU5R cannot be cancelled.",
        "Agent does not tell the user that reservation IFOYYZ has been cancelled.",
        "Agent does not tell the user that reservation NQNU5R has been cancelled.",
        "Agent does not tell the user that reservation M20IZO has been modified.",
    ),
    "10": (
        "Agent tells the user that a business cabin upgrade must apply to all flights in reservation 4NQLHD, not only the outbound IAH to SEA flights.",
        "Agent tells the user that upgrading reservation 4NQLHD to business for all passengers would cost more than $1000 in total.",
        "Agent does not offer to change the cabin for only some of the flights in reservation 4NQLHD.",
    ),
    "13": (
        "Agent tells the user that reservation XEWRD9 cannot be changed to a nonstop flight from ATL to Las Vegas.",
        "Agent calls the transfer_to_human_agents tool when the user asks to be transferred to a human agent.",
    ),
    "26": (
        "Agent tells the user that reservation 3FRNFB cannot be cancelled.",
        "Agent does not tell the user that reservation 3FRNFB has been cancelled.",
        "Agent does not offer the user a refund for reservation 3FRNFB.",
    ),
    "27": (
        "Agent verifies that flight HAT039 from ATL to SEA on 2024-05-15 was delayed.",
        "Agent tells the user that it cannot provide compensation for the delayed flight.",
        "Agent does not tell the user that a travel certificate has been issued.",
    ),
    "28": (
        "Agent tells the user that reservation SI5UKW cannot be cancelled.",
        "Agent does not tell the user that reservation SI5UKW has been cancelled.",
        "Agent does not offer the user a refund, full or partial, for reservation SI5UKW.",
        "Agent does not offer the user a travel voucher, gift card, or certificate as compensation.",
    ),
    "31": (
        "Agent tells the user that the flights in reservation DF89BM cannot be changed.",
        "Agent does not tell the user that the flights in reservation DF89BM have been changed.",
        "Agent does not tell the user that a new reservation has been booked.",
    ),
    "34": (
        "Agent tells the user that the requested changes to reservation HXDUBJ would cost more than $200.",
        "Agent does not tell the user that reservation HXDUBJ has been modified.",
    ),
    "36": (
        "Agent looks up the details of reservation EUJUY6.",
        "Agent tells the user that the flights in reservation EUJUY6 cannot be changed.",
        "Agent does not tell the user that the flights in reservation EUJUY6 have been changed.",
    ),
    "38": (
        "Agent verifies that flight HAT018 on 2024-05-11 was delayed.",
        "Agent determines from the reservation record that reservation 4OG6T3 has 1 passenger.",
        "Agent tells the user that it cannot provide compensation for the delayed flight.",
        "Agent does not tell the user that a travel certificate has been issued.",
    ),
    "41": (
        "Agent looks up the details of reservation 8C8K4E.",
        "Agent looks up the details of reservation UDMOP1.",
        "Agent looks up the details of reservation XAZ3C0.",
        "Agent looks up the details of reservation LU15PA.",
        "Agent looks up the details of reservation MSJ4OA.",
        "Agent looks up the details of reservation I6M8JQ.",
        "Agent looks up the details of reservation 4XGCCM.",
        "Agent tells the user that reservation UDMOP1 cannot be cancelled.",
        "Agent does not tell the user that any of the seven reservations has been cancelled.",
    ),
    "43": (
        "Agent tells the user that reservation D1EW9B cannot be cancelled.",
        "Agent tells the user that reservation 9HBUV8 cannot be cancelled.",
        "Agent does not tell the user that reservation D1EW9B has been cancelled.",
        "Agent does not tell the user that reservation 9HBUV8 has been cancelled.",
    ),
    "45": (
        "Agent tells the user that reservation PEP4E0 cannot be cancelled.",
        "Agent tells the user that the flights in reservation PEP4E0 cannot be changed.",
        "Agent does not tell the user that reservation PEP4E0 has been cancelled.",
        "Agent does not tell the user that the flights in reservation PEP4E0 have been changed.",
        "Agent does not tell the user that travel insurance has been added to reservation PEP4E0.",
    ),
    "46": (
        "Agent tells the user that the travel insurance on reservation H8Q05L cannot be refunded.",
        "Agent does not tell the user that the travel insurance on reservation H8Q05L can be removed.",
        "Agent does not offer the user any refund, certificate, or other compensation.",
        "Agent does not tell the user that the travel insurance on reservation H8Q05L has been removed.",
    ),
    "47": (
        "Agent looks up the details of reservation H8Q05L.",
        "Agent tells the user that reservation H8Q05L cannot be cancelled.",
        "Agent does not tell the user that reservation H8Q05L has been cancelled.",
    ),
    "48": (
        "Agent looks up the details of reservation 3RK2T9.",
        "Agent tells the user that reservation 3RK2T9 was booked more than 24 hours ago.",
        "Agent tells the user that reservation 3RK2T9 cannot be cancelled.",
        "Agent does not tell the user that reservation 3RK2T9 has been cancelled.",
    ),
    "49": (
        "Agent looks up the details of reservation 3RK2T9.",
        "Agent tells the user that reservation 3RK2T9 does not include travel insurance.",
        "Agent tells the user that reservation 3RK2T9 cannot be cancelled.",
        "Agent does not tell the user that reservation 3RK2T9 has been cancelled.",
    ),
}


def assertions_for_task(domain: str, task_split: str, task_id: str) -> tuple[str, ...] | None:
    """Return the adopted assertions only on their audited evaluation split."""
    if domain != "airline" or task_split != "base":
        return None
    return ADOPTED_NL_ASSERTIONS.get(str(task_id))
