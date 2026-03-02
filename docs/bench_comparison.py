#!/usr/bin/env python3
"""Generate grouped bar chart comparing CPU vs Metal branches and configurations."""

import matplotlib.pyplot as plt
import numpy as np

degrees = ['d13', 'd14', 'd15', 'd16', 'd17']

# prove_min values from benchmarks (in ms)
cpu_only           = [277,  541,  1107, 2284, 5815]
old_merkle         = [237,  455,   897, 1897, 5555]  # feature/metal-poseidon branch
new_merkle         = [227,  464,   917, 1937, 4916]  # feature/metal-ntt-coalesced (Merkle only)
new_merkle_ntt     = [231,  468,   990, 2186, 5098]  # feature/metal-ntt-coalesced (Merkle + NTT)

x = np.arange(len(degrees))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - 1.5*width, cpu_only,       width, label='CPU Only',                          color='#4A90D9')
bars2 = ax.bar(x - 0.5*width, old_merkle,     width, label='Old Merkle (metal-poseidon)',       color='#C084FC')
bars3 = ax.bar(x + 0.5*width, new_merkle,     width, label='New Merkle (ntt-coalesced)',        color='#E8943A')
bars4 = ax.bar(x + 1.5*width, new_merkle_ntt, width, label='New Merkle + NTT (ntt-coalesced)',  color='#6AB04C')

ax.set_xlabel('Degree', fontsize=13)
ax.set_ylabel('Prove Time (ms)', fontsize=13)
ax.set_title('E2E Prove Time: CPU vs Old Merkle vs New Merkle vs New Merkle+NTT', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(degrees, fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

# Add speedup vs CPU annotations
for i in range(len(degrees)):
    for bars, vals, color in [
        (bars2, old_merkle, '#C084FC'),
        (bars3, new_merkle, '#E8943A'),
        (bars4, new_merkle_ntt, '#6AB04C'),
    ]:
        speedup = cpu_only[i] / vals[i]
        bar = bars[i]
        ax.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, vals[i]),
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', fontsize=7, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/fujie/coding/cysic/20260220/plonky2-fork/docs/bench_comparison.png', dpi=150)
print('Saved to docs/bench_comparison.png')
