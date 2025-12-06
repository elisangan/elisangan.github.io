import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import json
import os

# LOAD DATA
with open('AOSC211_met-metadata.json', encoding='utf-8') as f_raw:
    raw_data = json.load(f_raw)
with open('AOSC211_mask-metadata.json', encoding='utf-8') as f_crop:
    crop_data = json.load(f_crop)

img_dir = '\POSE_CLUSTERS\00_RAW'

# FILTER OUT HEADS AND BUSTS
raw_ids = {str(obj['objectID']): obj for obj in raw_data}
exclude_types = {'Head', 'Bust', 'Bust fragment', 'Hand', 'Hands', 'Fragment', 'Sculpture fragment'}

crop_data = [
  obj for obj in crop_data
  if raw_ids.get(obj['objectID'], {}).get('objectName', '') not in exclude_types
]

print(len(crop_data))

# PUT IN DATAFRAME
df_raw = pd.DataFrame(raw_data)
df_crop = pd.DataFrame(crop_data)

# VIEW METADATA
# print(df_raw.head().to_string())
print(df_crop.head().to_string())

# VIEW SCHEMA
print(df_raw.columns.tolist())
for col in df_raw.columns:
    print(f"| {col} | {df_raw[col].dtype} |")
print(df_crop.columns.tolist())
for col in df_crop.columns:
    print(f"| {col} | {df_crop[col].dtype} |")


# -------------------------------------------------------------------------------------

# K-MEANS FUNCTIONS

def initialize_centroids(data, K):

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    n = data.shape[1]

    centers = np.random.rand(K , n) * (data_max - data_min) + data_min
    shape = np.array(centers)

    return shape

def get_label(data, centers):

    distances = np.zeros((len(data), len(centers)))
    for i, x in enumerate(data):
      for j, mu in enumerate(centers):
        distances[i,j] = np.linalg.norm(x - mu)

    labels = np.argmin(distances, axis = 1)

    return labels

def move_centroids(data, K, labels):

    n = data.shape[1]
    new_centers = np.zeros((K, n))

    # for i in np.arange(K):
    #     cluster_K = data[labels == i]
    #     new_center = np.mean(cluster_K, axis = 0)
    #     new_centers[i] = new_center

    for i in np.arange(K):
        cluster_K = data[labels == i]
        if len(cluster_K) == 0:
            new_centers[i] = data[np.random.randint(len(data))]
        else:
            new_centers[i] = np.mean(cluster_K, axis=0)

    return new_centers

def cluster_Kmeans(data, K, maxiteration = 1000):

    n = data.shape[1]
    previous_centers = np.zeros((K, n))
    updated_centers = initialize_centroids(data, K)
    count = 1

    while not np.all(updated_centers==previous_centers) and count < maxiteration:

        previous_centers = updated_centers
        labels = get_label(data, previous_centers)
        updated_centers = move_centroids(data, K, labels)
        count +=1

    print("iterations required: ", count)

    return updated_centers, labels


# -------------------------------------------------------------------------------------

# ASPECT RATIO

# EXTRACT WIDTH AND HEIGHT

widths = np.array([obj['bounding_box']['width'] for obj in crop_data])
heights = np.array([obj['bounding_box']['height'] for obj in crop_data])

# DERIVE ASPECT RATIO

aspect_ratios_raw = widths / heights

ar_min, ar_max = aspect_ratios_raw.min(), aspect_ratios_raw.max()
aspect_ratios_scaled = (aspect_ratios_raw - ar_min) / (ar_max - ar_min)



# -------------------------------------------------------------------------------------


# CENTER OF GRAVITY

def get_center_of_gravity(mask_path):

    img = Image.open(mask_path).convert('L')
    mask = np.array(img) > 0

    if mask.sum() == 0:
        return 0.5, 0.5

    y_coords, x_coords = np.where(mask)

    cog_x = x_coords.mean()
    cog_y = y_coords.mean()

    return cog_x, cog_y, mask.shape[1], mask.shape[0]

cog_y_raw = []
img_heights = []

for obj in crop_data:
  mask_path = os.path.join(img_dir, str(obj['objectID']), 'primary_mask.png')
  cog_x, cog_y, img_w, img_h = get_center_of_gravity(mask_path)
  cog_y_raw.append(cog_y)
  img_heights.append(img_h)

cog_y_raw = np.array(cog_y_raw)
img_heights = np.array(img_heights)

cog_y_scaled = cog_y_raw / img_heights



# -------------------------------------------------------------------------------------


# K MEANS RUN

features = np.column_stack((aspect_ratios_scaled, cog_y_scaled))

# # INDIVIDUAL K MEANS RUN

# K = 3
#
# mu_init = initialize_centroids(features, K)
# np.random.seed(42)
#
# print("initial centroids:", mu_init)
# print("shape of mu_init:", mu_init.shape)
#
# labels = get_label(features, mu_init)
# print("shape of labels:", labels.shape)
#
# mu_new = move_centroids(features, K, labels)
# print("new centroids:", mu_new)
# print("shape of mu_new:", mu_new.shape)
#
# new_centers, labels = cluster_Kmeans(features, K, maxiteration = 1000)

# BATCH K MEANS RUN

# K_runs = [3, 4, 5]
K_runs = [3, 7, 8]
results = {}

for K in K_runs:
    new_centers, labels = cluster_Kmeans(features, K, maxiteration = 1000)
    results[K] = {'centers': new_centers, 'labels': labels}


# -------------------------------------------------------------------------------------


# PLOTTING

plt.style.use('dark_background')
plt.rcParams['font.family'] = 'Crimson Pro'

# # PLOT FEATURES ON UNSCALED AND SCALED GRID
#
# y_min = cog_y_scaled.min()
# y_max = cog_y_scaled.max()
# padding = 0.05 * (y_max - y_min)
#
# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
#
# ax1.scatter(aspect_ratios_raw, cog_y_raw, color = '#bfcaff', alpha=0.7, edgecolors='none')
# ax1.set_title("Unscaled Features", fontsize=14, pad=20)
# ax1.set_xlabel("aspect ratio", fontsize=14, fontstyle='italic', labelpad=20, alpha=0.8)
# ax1.set_ylabel("vertical center of gravity", fontsize=14, fontstyle='italic', labelpad=20, alpha=0.8)
#
# ax2.scatter(aspect_ratios_scaled, cog_y_scaled, color = '#bfcaff', alpha=0.7, edgecolors='none')
# ax2.set_title("Scaled Features", fontsize=14, pad=20)
# ax2.set_xlabel("aspect ratio", fontsize=14, fontstyle='italic', labelpad=20, alpha=0.8)
# ax2.set_ylabel("vertical center of gravity", fontsize=14, fontstyle='italic', labelpad=20, alpha=0.8)
# ax2.set_xlim(0,1)
# ax2.set_ylim(y_min - padding, y_max + padding)
#
# for ax in [ax1, ax2]:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.grid(True, color='#999999', linewidth=0.5, alpha=0.3)
#
# plt.tight_layout(pad=3.0)
# plt.subplots_adjust(wspace=0.3)
# plt.savefig('0_1_unscaled_scaled_aspect_ratio_cog_y.png', dpi=300, facecolor='black')
# plt.show()

# # PLOT INITIAL CENTROIDS
#
# fig2 = plt.figure(figsize=(8, 8))
#
# plt.scatter(features[:,0], features[:,1], c='#bfcaff')
# plt.scatter(mu_init[:,0], mu_init[:,1], c='#6f0', marker='+', s=200)
# plt.title("Initial Centroids")
# plt.xlabel("width", fontstyle='italic')
# plt.ylabel("height", fontstyle='italic')
#
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.grid(True, color='#999999', linewidth=0.5, alpha=0.3)
#
# plt.tight_layout(pad=3.0)
# plt.subplots_adjust(wspace=0.3)
# plt.savefig('2_initial_centroids.png', dpi=300, facecolor='black')
# plt.show()
#
# # PLOT LABELS
#
# fig3 = plt.figure(figsize=(8, 8))
# plt.scatter(features[:,0], features[:,1], c=labels, cmap='cool', vmin=0, vmax=K-1, alpha=0.5)
# plt.scatter(mu_init[:, 0], mu_init[:, 1], c=np.arange(len(mu_init)), cmap='cool', vmin=0, vmax=K-1, marker='+', s=200)
#
# plt.title("Initial Label")
# plt.xlabel("aspect ratio", fontstyle='italic')
# plt.ylabel("center of gravity y", fontstyle='italic')
#
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.grid(True, color='#999999', linewidth=0.5, alpha=0.3)
#
# plt.tight_layout(pad=3.0)
# plt.subplots_adjust(wspace=0.3)
# plt.savefig('3_initial_labels.png', dpi=300, facecolor='black')
# plt.show()
#
# # PLOT INITIAL CLUSTERS
#
# fig4 = plt.figure(figsize=(8, 8))
# plt.scatter(features[:,0], features[:,1], c=labels, cmap='cool', vmin=0, vmax=K-1, alpha=0.5)
# plt.scatter(mu_new[:, 0], mu_new[:, 1], c=np.arange(len(mu_new)), cmap='cool', vmin=0, vmax=K-1, marker='+', s=200)
#
# plt.title("Initial Clusters")
# plt.xlabel("aspect ratio", fontstyle='italic')
# plt.ylabel("center of gravity y", fontstyle='italic')
#
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.grid(True, color='#999999', linewidth=0.5, alpha=0.3)
#
# plt.tight_layout(pad=3.0)
# plt.subplots_adjust(wspace=0.3)
# plt.savefig('4_initial_clusters.png', dpi=300, facecolor='black')
#
# plt.show()
#
# # PLOT FULL K-MEANS
#
# fig5 = plt.figure(figsize=(8, 8))
# plt.scatter(features[:,0], features[:,1], c=labels, cmap='cool', vmin=0, vmax=K-1, alpha=0.3)
# plt.scatter(new_centers[:, 0], new_centers[:, 1], c=np.arange(len(new_centers)), cmap='cool', vmin=0, vmax=K-1, marker='+', s=200)
#
# plt.title("K Means Three Clusters")
# plt.xlabel("aspect ratio", fontstyle='italic')
# plt.ylabel("center of gravity y", fontstyle='italic')
#
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.grid(True, color='#999999', linewidth=0.5, alpha=0.3)
#
# plt.tight_layout(pad=3.0)
# plt.subplots_adjust(wspace=0.3)
# plt.savefig('5A_K-Means-3.png', dpi=300, facecolor='black')
#
# plt.show()

# K COMPARISONS

fig6, axes = plt.subplots(1, 3, figsize=(24, 8))

for idx, K in enumerate(K_runs):
    ax = axes[idx]
    centers = results[K]['centers']
    labels = results[K]['labels']

    scatter = ax.scatter(features[:,0], features[:,1], c=labels, cmap='cool', vmin=0, vmax=K-1, alpha=0.3)
    ax.scatter(centers[:, 0], centers[:, 1], c=np.arange(K), cmap='cool', vmin=0, vmax=K-1, marker='+', s=200)

    for i, (cx, cy) in enumerate(centers):
        count = np.sum(labels == i)
        ax.annotate(f'{count} pts', (cx, cy), xytext=(0,0), textcoords='offset points', color='white')

    ax.set_title(f"K = {K} Clusters")
    ax.set_xlabel("aspect ratio", fontstyle='italic')
    ax.set_ylabel("center of gravity y", fontstyle='italic')

    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
    axes[idx].spines['bottom'].set_visible(False)
    axes[idx].spines['left'].set_visible(False)
    axes[idx].grid(True, color='#999999', linewidth=0.5, alpha=0.3)

plt.tight_layout(pad=3.0)
plt.subplots_adjust(wspace=0.3)
plt.savefig('6_K-Means-Comparisons-2.png', dpi=300, facecolor='black')

plt.show()


# UNPACKED CLUSTER PLOT

K = 7
centers = results[K]['centers']
labels = results[K]['labels']

cmap = plt.cm.cool
colors = [cmap(i / (K - 1)) for i in range(K)]

fig, axes = plt.subplots(K, 1, figsize=(24, K*1.25))

for cluster_id in range(K):
    ax = axes[cluster_id]

    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    x_values = features[cluster_indices, 0]

    sorted_order = np.argsort(x_values)
    sorted_indices = cluster_indices[sorted_order]

    for pos, idx in enumerate(sorted_indices):

        img_path = os.path.join(img_dir, str(crop_data[idx]['objectID']), 'cropped_black.png')
        img = Image.open(img_path)
        img.thumbnail((50, 50))
        im = OffsetImage(np.array(img), zoom=0.5)

        x_val = features[idx, 0]
        x_pos = pos / max(len(sorted_indices) - 1, 1)

        ab = AnnotationBbox(im, (x_pos, 0), frameon=True, bboxprops=dict(edgecolor=colors[cluster_id], linewidth=1, alpha=0.2, facecolor='none'))
        ax.add_artist(ab)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1, 1)
    ax.set_title(f"Cluster {cluster_id}, {len(sorted_indices)} Sculptures", fontsize=14)
    ax.set_yticks([])
    ax.axis('off')

plt.tight_layout()
plt.savefig('8_K7_cluster_rows.png', dpi=300, facecolor='black')
plt.show()

