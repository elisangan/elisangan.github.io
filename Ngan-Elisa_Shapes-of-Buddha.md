# The Shapes of Buddha

Elisa Ngan <br>
Design Media Arts MFA <p>

Dr. Alexander Lozinski <br>
Atmospheric & Oceanic Sciences C204, Fall 2025 <p>

## Introduction

Buddhist art describe the life of the Buddha and the cast of characters surrounding him.
In its communicative role, Buddhist art artifacts exhibit iconographic consistency such that followers can recognize scenes, identify characters, and understand thematic ideas. 
The Buddha for example can be identified through his long earlobes, the curls of his hair, and his flowing robe. 
The consistency and longitudinal continuity of features across Buddhist art artifacts makes it a particularly promising area in which to apply machine learning techniques.
This paper explores the potential of these machine learning techniques by applying the K-means algorithm to cluster and identify common poses found in Cambodian Buddhist sculptures.

## Data Collection & Exploration

The dataset was collected from the Metropolitan Museum of Art through their API after exploring the search results displayed on its website.
Search terms used to capture the full idea of "Cambodian Buddhist sculptures" included "Cambodia," "Buddhist Sculptures," "Khmer," and "Angkor."
The website was scraped with the SCRAPY library in order to obtain the `object_id` of target artifacts retrieved through the search terms. 
`object_id` scraped from the website was then used to pull matching images and all related metadata from the API. 
A total of 576 objects representing the full union of the search terms was downloaded onto a machine to be processed locally.

Exploration of the images confirmed that there were patterns to the poses of the figures. 
The most immediately evident Buddha poses were: 1) standing, 2) sitting, and 3) reclining.
However, while the base class of the dataset had been limited to sculptures, there were other unexpected objects that were included such as architectural elements.
Refining the dataset further to include only single figures that could be rotated 360-degrees required excluding additional object types and tags.
After filtering further through the metadata structure, the dataset was culled to 309 objects.
The table below describes the running total of the dataset from download to final.

  | Stage | Description            | Count | Removed | Filter Details                    |
  |-------|------------------------|-------|---------|-----------------------------------|
  | 0     | Raw Download (Met API) | 576   | —       | Sculpture Classification query    |
  | 1     | Image Processing       | 539   | 37      | Missing images                    |
  | 2     | Object Type Filter     | 381   | 158     | 54 excluded object types          |
  | 3     | Tag Filter             | 377   | 4       | 26 excluded tags                  |
  | 4     | Fragment Filter        | 309   | 85      | 7 excluded tags                   |
  | Final | Filtered Dataset       | 292   |         |                                   |

> <b>Excluded Object Types</b><br>
> * Architectural elements and reliefs (Altarpiece, Antefix, Architectural element, Architectural
   relief, Column, Frieze section, Lintel, Pedestal, Pillar fragment, Relief, Relief Panel,
  Relief fragment, Relief panel, Stair riser)
> * Religious/ritual objects (Buddhist triad, Censer, Linga, Reliquary, Rondel, Shrine, Shrine panel, Stele, Stele and base, Stele
  fragment, Stele section, Stupa, Stupa model, Votive plaque, Votive tablet)
> * Decorative elements (Crown section, Finial, Finial and chime, Garland holder, Pagoda, Pagoda base, Roof
  finial, Tile)
> * Functional objects (Container, Dagger, Mold, Mold and impression, Palanquin
  ring, Plaque, Plaques, Rock crystal, Sealing)
> * Fragment objects (Head, Fragment, Bust, Hand, Sculpture fragment, Bust fragment, Hands)
> * Other categories (Base of stele, Drum slab, Figures, Group, Relic, Temple model).

 > <b>Excluded Tags</b><p>
 > Antefix, Architectural Element, Column, Dagger, Fragment, Frieze, Garland Holder, Incense
  Burner, Linga, Lintel, Lions, Model, Pagoda, Panel, Pilaster, Pillar, Plaque, Relief,
  Reliquary, Ring, Rondel, Stair Riser, Stele, Stupas, Tablet, Tile

After filtering the dataset to the desired objects representing "Cambodian Buddhist Sculptures," the metadata downloaded from the API was explored to understand what features could be used to classify poses in Euclidean space.
It was clear that the Met did not think of these images as images since the schema primarily described the sculptures.
Features would have to be engineered.

## Feature Engineering
Aspect ratio and vertical weight distribution were designed for use in the K-means algorithm based on the hypothesis that they would be useful for pose classification.

### Data Cleaning
The images were photographed in inconsistent and heterogeneous ways. This creates noise since the distance between the figure and the camera was not consistent and normalized along the z-axis.
In order to resolve this problem, the figure was isolated from the background such that the extents of the figure can represent pose more closely without extraneous information.
Understanding what the figure is in the image requires the grid of pixels in the image itself to be classified as either figure / not-figure or background / not-background.
In order to do this, a computer vision technique called image segmentation was used where pixels are assigned and grouped into segments to produce binary masks.
White pixels in the mask are part of the figure / background representation while black pixels are not.
Since one binary mask represents the background and another binary mask represents the figure and both correspond in coordinate space, the background can be separated from the figure such that the figure can be isolated for further data extraction. 

### Aspect Ratio
The extents of the white pixels in the binary mask representing the figure was used to extract the `height` and `width` of the image in order to calculate its `aspect ratio`.
The width and the height of the figure was hypothesized to be promising for classification because the aspect ratio of the figure can help clearly determine whether the figure is standing or reclining.
If the figure is taller than wide, then the figure is standing; if the figure is wider than it is tall, then the figure is reclining. 
However, if the width and height are roughly equal, it might be assumed that the figure is sitting but a close look at the dataset shows that some figures are sitting on pedestals which may vertically elongate the aspect ratio to make it look like they are standing.

### Vertical Weight Distribution
In order to help disambiguate between standing and sitting images, the vertical weight distribution of the figure was calculated.
Using the binary masks, rows of white pixels along the height of the object were averaged to understand the distribution of white pixels across its height.
Sitting figures will have a lower vertical weight distribution with more white pixels towards the bottom and less white pixels towards the top.
The vertical weight distribution translates into the vertical center of gravity, `cog y`, for an image where a lower center of gravity indicate the possibiblity of a sitting pose.

## Data Processing

A custom data platform integrating Meta's SAM vit-b, built by the author for this project, was used to perform the image segmentation. 
The 377 images were first processed as a batch to produce binary masks.
Connectivity and area metrics were then used to automatically identify whether an image was background or figure.
The masks were then reviewed to ensure that they correctly captured the underlying figure.
If it did not, then -- through pixel-level operations of addition and subtraction -- several masks could be composited from the set that was generated to create a crafted mask.
If the masks generated in the first pass were of extremely poor quality, then the model's hyperparameters could be turned to re-process the individual image.

<b>Mask Generator</b>
<img width="2560" height="1416" alt="img" src="https://github.com/user-attachments/assets/1ff28de5-8ef3-4cb5-9907-d8ea8e81f6b1" />


159 figure masks were reviewed to be acceptable representations but 218 figure masks had to be edited and manually crafted.
This took an incredibly long time. Halfway through, data dropped and all work was lost.
After the data was cleaned, bounding boxes were extracted using the extents of the figure mask to extract the final `width` and `height`.
The bounding box was also used to crop the images down with 5% padding in order to normalize the depth of field by removing / deleting unnecessary background information.

## K-Means Implementation

Each object was plotted on a grid with its `aspect ratio` as its x-coordinate and `cog y` as its y-coordinate.
The values of each feature was then normalized from 0 to 1 and plotted again to check for distortions in transformation.
Values for center of gravity were narrow, so plots used the min and max extents in order to view the datapoints more clearly.<p>

<img width="4800" height="2400" alt="0_1_unscaled_scaled_aspect_ratio_cog_y" src="https://github.com/user-attachments/assets/fa98af73-41f4-45aa-bb76-13861afba25a" />

Finally, the K-Means algorithm was implemented to cluster the points into groups. 
Initialized cluster centroids and labels were plotted to confirm that datapoints were being assigned to the centroid labels and mu was converging into the average location of the cluster centroid as expected.

<img width="2400" height="2400" alt="2_initial_centroids" src="https://github.com/user-attachments/assets/2a8e0b66-93c4-40d8-adff-f1b0bf024dcb" />
<img width="2400" height="2400" alt="3_initial_labels" src="https://github.com/user-attachments/assets/062c7e0e-a8ae-4e41-b909-0add3e7ad624" />
<img width="2400" height="2400" alt="5A_K-Means-3" src="https://github.com/user-attachments/assets/7de4d76a-b490-4516-ba8b-448afbcef8d5" />


## K-Means Analysis & Verification

The reason why the K-means algorithm was used for classifying these figures was because the number of common Buddha poses is assumed to be known at three: 1) standing, 2) sitting, and 3) reclining.
When K was set to 3 and the model applied, all three poses were successfully grouped together.
There were 123 sitting sculptures were assigned to cluster 0, and 168 standing sculptures assigned to cluster 1.
There was only one reclining Buddha in its own cluster 2.
The density of the points in each cluster illustrates the iconographic consistency of these poses.
In order to verify the results more closely, a new plot was created to plot the figures in each cluster in order of their position along the x-axis as the aspect ratios if the figure extent got wider.
When the three cluster analysis was unpacked, when cluster 1 containing the standing sculptures moved across the x-axis and aspect ratios got wider, more sitting figures started to be included, demonstrating that the intent of `cog_y` and the vertical weight distribution was working exactly as intended.

<img width="7200" height="2400" alt="6_K-Means-Comparisons-1" src="https://github.com/user-attachments/assets/3200a0b3-5c8b-4ae6-8844-95b66729007d" />
<img width="7200" height="1125" alt="8_K3_cluster_rows" src="https://github.com/user-attachments/assets/dbce681c-c443-4a99-9ae8-3734ab05e86b" /><p>


Increasing cluster size  resulted in clusters with more nuance in shapes. 
Shown below are sculpture assignments at different values for k.
The nuanced but clear differences in shapes that can be observed between groups illustrates the success of the features engineered for this k-means classification.
While each artifact was undoubtedbly unique, different canonical shapes still emerged as different numbers of clusters were initialized.
Although the initial pose classification goals were to identify the three poses, as K increased beyond three, new patterns emerged that provided more insights about the dataset.
For example, when more clusters are added, it became clear that sometimes the Buddha sits on the ground, his silouhette creating a sharp equilateral triangle , but at other times the Buddha sits on a slightly sloping pedestal, his silouhette with the pedestal creating a more vertical isoceles triangle. Sometimes individual legs can clearly be seen; othertimes there is the rectangular box created by a flowing robe.

<img width="7200" height="4800" alt="6_K-Means-Comparisons-2" src="https://github.com/user-attachments/assets/e16f1d55-d9c2-486f-8a69-9ad5dc9f3e34" />
<img width="7200" height="2250" alt="8_K6_cluster_rows" src="https://github.com/user-attachments/assets/961f176b-c38e-41c0-bfe8-c2c424517c1e" />
<img width="7200" height="2625" alt="8_K7_cluster_rows" src="https://github.com/user-attachments/assets/d9d03cb8-d994-4def-8e71-1f494b854646" />
<img width="7200" height="3000" alt="8_K8_cluster_rows" src="https://github.com/user-attachments/assets/5979ae49-9f06-4fe7-8adc-29dc68d4948a" /><p>


That the aspect ratio and vertical weight distribution is able to capture and classify the range of Buddha sculptures in perceptibly clear ways illustrates the regularity and consistency of Buddhist art iconography.
This is quite astounding considering that the earliest / oldest artifact is dated to Year 0 (1st century) and the latest artifact is dated to 1999.
In total the artifacts in this dataset have been depicting and telling the same story about one man for roughly 2000 years.

| Century | Count |
|---------|-------|
| 1st CE  | 17    |
| 2nd CE  | 2     |
| 3rd CE  | 6     |
| 4th CE  | 4     |
| 5th CE  | 11    |
| 6th CE  | 19    |
| 7th CE  | 31    |
| 8th CE  | 29    |
| 9th CE  | 28    |
| 10th CE | 27    |
| 11th CE | 17    |
| 12th CE | 28    |
| 13th CE | 7     |
| 14th CE | 9     |
| 15th CE | 14    |
| 16th CE | 8     |
| 17th CE | 11    |
| 18th CE | 12    |
| 19th CE | 6     |
| 20th CE | 6     |
| Total   | 292   |

## Next Steps

The dataset, despite being heavily cleaned, still had stray head fragments.
Tagging and filtering them out through the custom data platform in the future would result in cleaner and more satisfying results.

Adding to the base dataset by including sculptures from other museums would make this a more comprehensive resource for those interested in learning more about buddhist statues.

Finally, now that the images have been classified, they can be easily labeled and used to train a custom pose classifier that can then be used to predict poses for additional images.
