import json
import matplotlib.pyplot as plt
import numpy as np

# Загружаем результаты
with open('runs_cpu/results.json', 'r') as f:
    data = json.load(f)

epochs = list(range(1, 6))
local_loss = data['local']['loss']
ig_loss = data['ig']['loss']
ig_energy = data['ig']['energy']

# График
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Loss curves
ax1.plot(epochs, local_loss, 'o-', linewidth=2, markersize=8, 
         label='Local', color='#2E86AB')
ax1.plot(epochs, ig_loss, 's-', linewidth=2, markersize=8, 
         label='IG (α=1.4)', color='#A23B72')

# Аннотации финальной разницы
final_delta = ((ig_loss[-1] - local_loss[-1]) / local_loss[-1]) * 100
ax1.text(5, ig_loss[-1] + 0.002, f'+{final_delta:.1f}%', 
         fontsize=11, color='#A23B72', weight='bold',
         ha='center')

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Training Loss Comparison', fontsize=14, weight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.045, 0.062])

# Subplot 2: Относительная разница
delta_pct = [((ig - loc) / loc) * 100 for ig, loc in zip(ig_loss, local_loss)]
colors = ['green' if d < 0 else 'red' for d in delta_pct]

ax2.bar(epochs, delta_pct, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('IG vs Local (%)', fontsize=12)
ax2.set_title('Relative Difference (negative = IG better)', fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Аннотации
for i, (ep, d) in enumerate(zip(epochs, delta_pct)):
    ax2.text(ep, d + (0.3 if d > 0 else -0.3), f'{d:+.1f}%', 
             ha='center', va='bottom' if d > 0 else 'top',
             fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig('runs_cpu/analysis_5epochs.png', dpi=150, bbox_inches='tight')
print('✓ Saved to runs_cpu/analysis_5epochs.png')

# Статистика
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Local final loss:  {local_loss[-1]:.6f}")
print(f"IG final loss:     {ig_loss[-1]:.6f}")
print(f"Final difference:  {final_delta:+.2f}%")
print(f"\nIG better epochs:  {sum(1 for d in delta_pct if d < 0)}/5")
print(f"Average delta:     {np.mean(delta_pct):+.2f}%")
print(f"Std delta:         {np.std(delta_pct):.2f}%")
print(f"\nEnergy (constant): {ig_energy[0]:.6f}")
print(f"Energy std:        {np.std(ig_energy):.8f} (should be ~0)")
print("="*60)

# Decision
print("\n🎯 DECISION CRITERIA:")
print("-" * 60)
if abs(final_delta) < 1.0:
    print("✅ Inconclusive (~equal performance)")
    print("   → Try different task or parameters")
elif final_delta < -2.0:
    print("✅ IG significantly better!")
    print("   → Continue research")
else:
    print("❌ IG worse by {:.1f}%".format(final_delta))
    print("   → MI-attention not suitable for time series")
    print("   → Try NLP tasks (semantic long-range dependencies)")

plt.show()