"""
Official implementation of Dynamic boundingboxes.

This module provides the Boundingbox class, designed to extract meaningful 
regions of interest from heatmaps using iterative peak detection, box merging, 
and Non-Maximal Suppression (NMS).
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
import matplotlib.pyplot as P
import matplotlib.patches as patches

# ==============================================================================
# CORE ANALYSIS CLASS
# ==============================================================================

class Boundingbox:
    """
    Analyzes heatmaps by identifying local maxima and generating bounding boxes.

    The process involves:
    1. Finding the highest values in a heatmap.
    2. Defining square grids around these points.
    3. Merging or suppressing overlapping grids to isolate distinct features.

    Attributes:
        heatmap: The 2D numpy array representing the density/intensity map.
        grid_fraction: Proportion of the heatmap dimension used for grid size.
        grid_size: Absolute pixel size of the square grid side.
        half_grid_size: Radius of the grid from the center point.
        grid_centers: List of tuples (row, col) representing peak locations.
        merged_boxes: List of lists [x1, y1, x2, y2] representing final areas.
    """

    def __init__(self, heatmap, grid_fraction=1/8):
        """
        Initializes the analyzer with a heatmap and calculates grid dimensions.

        Args:
            heatmap: numpy.ndarray input to be processed.
            grid_fraction: float, the ratio used to calculate grid_size 
                relative to heatmap height. Defaults to 0.125 (1/8).
        """
        self.heatmap = heatmap
        self.grid_fraction = grid_fraction
        # Grid size is determined by the number of rows (shape[0])
        self.grid_size = int(heatmap.shape[0] * grid_fraction)
        self.half_grid_size = self.grid_size // 2
        self.grid_centers = []
        self.merged_boxes = []

    def find_highest_points_with_grids(self, num_grids=10):
        """
        Iteratively finds the highest values in the heatmap.

        For each iteration, it finds the global maximum, records the center, 
        and then masks that area to -infinity to find the next distinct peak.

        Args:
            num_grids: int, maximum number of peak points to identify.

        Returns:
            list of dict: Each dict contains 'x' (column) and 'y' (row) 
                coordinates of a peak.

        Notes:
            - Modifies a copy of the heatmap internally to mask found regions.
            - Stops early if all remaining points are -inf.
        """
        processed_heatmap = np.copy(self.heatmap)

        for _ in range(num_grids):
            # Locate the global maximum index in the current state of the map
            max_index = np.unravel_index(np.argmax(processed_heatmap), processed_heatmap.shape)
            max_value = processed_heatmap[max_index]

            # Stop if no valid points are left (e.g., all points masked or below threshold)
            if max_value == -np.inf:
                break

            self.grid_centers.append(max_index)

            # Mask the surrounding area to prevent overlapping detections in this pass
            x, y = max_index # x = row index, y = column index
            x_start = max(0, x - self.half_grid_size)
            x_end = min(processed_heatmap.shape[0], x + self.half_grid_size + 1)
            y_start = max(0, y - self.half_grid_size)
            y_end = min(processed_heatmap.shape[1], y + self.half_grid_size + 1)
            
            # Apply the mask
            processed_heatmap[x_start:x_end, y_start:y_end] = -np.inf

        # Map back to standard Cartesian (x=col, y=row) for external consumption
        grid_centers_json = [{'x': center[1], 'y': center[0]} for center in self.grid_centers]
        return grid_centers_json

    def merge_overlapping_boxes(self):
        """
        Merges intersecting grid boxes into larger encompassing rectangles.

        Iterates through detected grid centers and combines any boxes that 
        touch or overlap with previously established merged boxes.

        Returns:
            list of dict: List of merged bounding boxes with 'x1', 'y1', 
                'x2', and 'y2' coordinates.

        Notes:
            - This method performs a greedy, single-pass merge.
            - The box format is [X_min, Y_min, X_max, Y_max].
        """
        self.merged_boxes = []
        for center in self.grid_centers:
            # center[0] is Row (Y), center[1] is Column (X)
            x, y = center
            # new_box format: [X1, Y1, X2, Y2]
            new_box = [y - self.half_grid_size, x - self.half_grid_size, y + self.half_grid_size, x + self.half_grid_size]
            merged = False

            # Check for overlaps with existing boxes and expand existing box to fit new_box
            for box in self.merged_boxes:
                # Check for non-overlap condition; if false, boxes overlap
                if not (new_box[2] < box[0] or new_box[0] > box[2] or new_box[3] < box[1] or new_box[1] > box[3]):
                    box[0] = min(box[0], new_box[0])
                    box[1] = min(box[1], new_box[1])
                    box[2] = max(box[2], new_box[2])
                    box[3] = max(box[3], new_box[3])
                    merged = True
                    break

            # If no intersection found, start a new merged box category
            if not merged:
                self.merged_boxes.append(new_box)

        merged_boxes_json = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]} for box in self.merged_boxes]
        return merged_boxes_json

    def non_max_suppression(self, threshold=0.25):
        """
        Refines boxes using Non-Maximal Suppression (NMS).

        Sorts boxes by area and removes those that have an Intersection 
        over Union (IoU) greater than the specified threshold with 
        a larger box.

        Args:
            threshold: float, IoU limit for suppressing overlapping boxes. 
                Higher values allow more overlap. Defaults to 0.25.

        Returns:
            list of dict: List of remaining bounding boxes after suppression.

        Notes:
            - Internally calls merge_overlapping_boxes() first to establish candidates.
            - Sorting is based on area (descending).
        """
        if not self.grid_centers:
            return []

        # Ensure boxes are merged before applying NMS
        self.merge_overlapping_boxes()

        boxes = np.array(self.merged_boxes)

        # Extraction based on [X1, Y1, X2, Y2] format
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]

        # Calculate areas for sorting and IoU denominators
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        # Sort by area in descending order
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Intersection coordinates
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Dimensions of the intersection
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Ratio of overlap (Intersection over Union)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Only retain indices where IoU is within acceptable threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        # Update the instance state with suppressed boxes
        self.merged_boxes = boxes[keep].tolist()

        merged_boxes_json = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]} for box in self.merged_boxes]
        return merged_boxes_json

    def plot_heatmap_with_grids(self):
        """
        Visualizes the heatmap with the identified bounding boxes overlaid.

        Boxes are rendered as blue outlines using Matplotlib patches.

        Side Effects:
            - Generates a Matplotlib plot.
            - Adds a colorbar to indicate intensity values.

        Notes:
            - P.show() is currently commented out in the source.
            - Rectangles are drawn using (x, y, width, height) logic.
        """
        fig, ax = P.subplots()
        # Display the base heatmap using the 'hot' colormap
        cax = ax.imshow(self.heatmap, cmap='hot', interpolation='nearest')
        fig.colorbar(cax)

        # Draw each bounding box in the current merged_boxes list
        for box in self.merged_boxes:
            # patches.Rectangle((x_start, y_start), width, height)
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # P.show()

# ==============================================================================
# END OF FILE
# ==============================================================================