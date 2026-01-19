import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'Siamese Neural Network Architecture', 
        fontsize=18, fontweight='bold', ha='center')

# Input samples (top)
input1_y = 8.5
input2_y = 8.5
ax.add_patch(FancyBboxPatch((1, input1_y), 1.5, 0.6, boxstyle="round,pad=0.05", 
                            facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(1.75, input1_y+0.3, 'Sample 1\n(2048)', ha='center', va='center', fontsize=10, fontweight='bold')

ax.add_patch(FancyBboxPatch((13.5, input2_y), 1.5, 0.6, boxstyle="round,pad=0.05", 
                            facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(14.25, input2_y+0.3, 'Sample 2\n(2048)', ha='center', va='center', fontsize=10, fontweight='bold')

# Shared Embedding Network (Conv) - Left branch
conv_start_y = 7
ax.arrow(1.75, input1_y, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)

# Conv1D layer 1
ax.add_patch(Rectangle((1, conv_start_y-0.5), 1.5, 0.5, facecolor='#FFB6C1', edgecolor='black', linewidth=1.5))
ax.text(1.75, conv_start_y-0.25, 'Conv1d(1→64)\nk=7', ha='center', va='center', fontsize=8)
ax.arrow(1.75, conv_start_y-0.5, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

# LeakyReLU + MaxPool
ax.add_patch(Rectangle((1, conv_start_y-1.3), 1.5, 0.3, facecolor='#98FB98', edgecolor='black', linewidth=1.5))
ax.text(1.75, conv_start_y-1.15, 'LeakyReLU\nMaxPool(4)', ha='center', va='center', fontsize=7)
ax.arrow(1.75, conv_start_y-1.3, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

# Conv1D layer 2
ax.add_patch(Rectangle((1, conv_start_y-2.1), 1.5, 0.5, facecolor='#FFB6C1', edgecolor='black', linewidth=1.5))
ax.text(1.75, conv_start_y-1.85, 'Conv1d(64→256)\nk=7', ha='center', va='center', fontsize=8)
ax.arrow(1.75, conv_start_y-2.1, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

# LeakyReLU + MaxPool
ax.add_patch(Rectangle((1, conv_start_y-2.9), 1.5, 0.3, facecolor='#98FB98', edgecolor='black', linewidth=1.5))
ax.text(1.75, conv_start_y-2.75, 'LeakyReLU\nMaxPool(4)', ha='center', va='center', fontsize=7)
ax.arrow(1.75, conv_start_y-2.9, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

# Flatten + FC layers
ax.add_patch(Rectangle((1, conv_start_y-3.6), 1.5, 0.3, facecolor='#DDA0DD', edgecolor='black', linewidth=1.5))
ax.text(1.75, conv_start_y-3.45, 'Flatten', ha='center', va='center', fontsize=8)
ax.arrow(1.75, conv_start_y-3.6, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

ax.add_patch(Rectangle((1, conv_start_y-4.3), 1.5, 0.4, facecolor='#FFD700', edgecolor='black', linewidth=1.5))
ax.text(1.75, conv_start_y-4.1, 'Linear(32256→1024)\nReLU', ha='center', va='center', fontsize=8)
ax.arrow(1.75, conv_start_y-4.3, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

# Embedding output
ax.add_patch(FancyBboxPatch((1.1, conv_start_y-5.1), 1.3, 0.4, boxstyle="round,pad=0.05", 
                            facecolor='#87CEEB', edgecolor='black', linewidth=2))
ax.text(1.75, conv_start_y-4.9, 'Embedding\n(128-dim)', ha='center', va='center', fontsize=9, fontweight='bold')

# Right branch (shared weights indicator)
ax.arrow(14.25, input2_y, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)

# Conv1D layer 1
ax.add_patch(Rectangle((13.5, conv_start_y-0.5), 1.5, 0.5, facecolor='#FFB6C1', edgecolor='black', linewidth=1.5))
ax.text(14.25, conv_start_y-0.25, 'Conv1d(1→64)\nk=7', ha='center', va='center', fontsize=8)
ax.arrow(14.25, conv_start_y-0.5, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

ax.add_patch(Rectangle((13.5, conv_start_y-1.3), 1.5, 0.3, facecolor='#98FB98', edgecolor='black', linewidth=1.5))
ax.text(14.25, conv_start_y-1.15, 'LeakyReLU\nMaxPool(4)', ha='center', va='center', fontsize=7)
ax.arrow(14.25, conv_start_y-1.3, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

ax.add_patch(Rectangle((13.5, conv_start_y-2.1), 1.5, 0.5, facecolor='#FFB6C1', edgecolor='black', linewidth=1.5))
ax.text(14.25, conv_start_y-1.85, 'Conv1d(64→256)\nk=7', ha='center', va='center', fontsize=8)
ax.arrow(14.25, conv_start_y-2.1, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

ax.add_patch(Rectangle((13.5, conv_start_y-2.9), 1.5, 0.3, facecolor='#98FB98', edgecolor='black', linewidth=1.5))
ax.text(14.25, conv_start_y-2.75, 'LeakyReLU\nMaxPool(4)', ha='center', va='center', fontsize=7)
ax.arrow(14.25, conv_start_y-2.9, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

ax.add_patch(Rectangle((13.5, conv_start_y-3.6), 1.5, 0.3, facecolor='#DDA0DD', edgecolor='black', linewidth=1.5))
ax.text(14.25, conv_start_y-3.45, 'Flatten', ha='center', va='center', fontsize=8)
ax.arrow(14.25, conv_start_y-3.6, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

ax.add_patch(Rectangle((13.5, conv_start_y-4.3), 1.5, 0.4, facecolor='#FFD700', edgecolor='black', linewidth=1.5))
ax.text(14.25, conv_start_y-4.1, 'Linear(32256→1024)\nReLU', ha='center', va='center', fontsize=8)
ax.arrow(14.25, conv_start_y-4.3, 0, -0.3, head_width=0.1, head_length=0.05, fc='gray', ec='gray', linewidth=1)

ax.add_patch(FancyBboxPatch((13.6, conv_start_y-5.1), 1.3, 0.4, boxstyle="round,pad=0.05", 
                            facecolor='#87CEEB', edgecolor='black', linewidth=2))
ax.text(14.25, conv_start_y-4.9, 'Embedding\n(128-dim)', ha='center', va='center', fontsize=9, fontweight='bold')

# Shared weights annotation
ax.annotate('', xy=(13.5, conv_start_y-2.5), xytext=(2.5, conv_start_y-2.5),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2, linestyle='dashed'))
ax.text(8, conv_start_y-2.2, 'Shared Weights', ha='center', fontsize=10, 
        fontweight='bold', color='red', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

# Arrows to Similarity Head
ax.arrow(1.75, conv_start_y-5.1, 3, -0.6, head_width=0.15, head_length=0.15, fc='black', ec='black', linewidth=1.5)
ax.arrow(14.25, conv_start_y-5.1, -3, -0.6, head_width=0.15, head_length=0.15, fc='black', ec='black', linewidth=1.5)

# Similarity Head
similarity_y = 1
ax.add_patch(FancyBboxPatch((5, similarity_y), 6, 0.8, boxstyle="round,pad=0.1", 
                            facecolor='#FFA07A', edgecolor='black', linewidth=2))
ax.text(8, similarity_y+0.6, 'Similarity Head', ha='center', fontsize=11, fontweight='bold')
ax.text(8, similarity_y+0.3, 'L1 + L2 + Element-wise × + Cosine', ha='center', fontsize=8)
ax.text(8, similarity_y+0.05, 'Linear(131→128) → Dropout(0.5) → Linear(128→1)', ha='center', fontsize=7)

# Output
ax.arrow(8, similarity_y, 0, -0.3, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
ax.add_patch(FancyBboxPatch((7, 0.2), 2, 0.4, boxstyle="round,pad=0.05", 
                            facecolor='#90EE90', edgecolor='black', linewidth=2))
ax.text(8, 0.4, 'Similarity Score\n(0 or 1)', ha='center', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('siamese_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Architecture diagram saved as 'siamese_architecture.png'")
