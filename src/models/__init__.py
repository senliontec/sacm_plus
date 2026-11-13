"""Model implementations shared by the baseline zoo.

The SAM network implementation (Meta SAM code plus the project adapters) lives
in ``models.sam_network`` and is consumed directly by the baseline adapters
(for example ``sam_vit_l_unet``).  The former algorithm layer has been
removed; the proposed method is implemented under
``baseline_zoo.models.sacm_gcd`` and registered through ``baseline_zoo``.
"""
