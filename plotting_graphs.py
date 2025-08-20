import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from os.path import join
from matplotlib.patches import Rectangle


def save(fsave):
    if fsave is not None:
        plt.savefig(join("plots",f"{fsave}.png"))


def plot_directed_undirected(y,c_table,landmarks,kind_labels,normalize=True, data_name="Digits",figsize=(10,5),fontsize=10,colors=None,use_legend=True,fsave=None):
    fig, ax = plt.subplots(facecolor="white",figsize=figsize)
    labels = [y[landmark] for landmark in landmarks]
    labels = [y] + labels
    positions = []
    position_between = []
    width=2
    j=-width
    for i,label in enumerate(labels):
        if i%2==1:
            j+=width
        else:
            position_between.append(j-width/2)
        positions.append(j)
        j+=width
    position_between = [positions[0]] +position_between[1:]

    histo = np.array([np.histogram(label,bins=len(c_table),density=False)[0] for label in labels])
    bottom = np.zeros(histo.shape[0])
    for i,label in enumerate(c_table):
        if normalize:
            histo_weight = histo[:,i]/histo.sum(axis=1)
        else:
            histo_weight = histo[:,i]
        
        if colors is not None:
            p = ax.bar(np.array(positions),histo_weight,width=width,label=label,bottom=bottom,color=colors[i])
        else:
            p = ax.bar(np.array(positions),histo_weight,width=width,label=label,bottom=bottom)
        bottom +=histo_weight

    for pos in position_between[1:]:
        ax.axvline(x=pos,ymin=0,ymax=1, color='black', linestyle='--')
    #ax.set_xticks(position_between,labels)
    #ax.set_xticks(position_between)
    
    #TODO change this
    for i,position in enumerate(positions):
        if i !=0:
            if i %2==1:
                ax.text(position, 1, "undirected", ha='center', va='bottom', fontsize=fontsize)
            else:
                ax.text(position, 1, "directed", ha='center', va='bottom', fontsize=fontsize)

    ax.set_xticks(position_between)
    ax.set_xticklabels(["Original Distribution"] + kind_labels)
    ax.set_xlim(positions[0]-width,positions[-1]+2*width)
    if normalize:
        ax.set_yticks(np.arange(start=0,stop=1.1,step=0.2))
        ax.set_yticklabels(np.arange(start=0,stop=110,step=20))
        ax.set_ylabel("Percentage in %")
    else: 
        ax.set_ylabel("Counts")

    ax.set_title(f"Landmark distribution on {data_name}")
    #ax.legend(loc="upper right",title="Classes")
    if use_legend:
        ax.legend(title="Classes",loc="center right")
    for spine in ax.spines.values():
        spine.set_visible(False)  # Hide all spines
    ax.spines.left.set_visible(True)
    ax.spines.left.set_bounds(0,1)
    ax.spines.bottom.set_visible(True)
    ax.spines.bottom.set_bounds(-width+positions[0],positions[-1]+width)
    save(fsave)
    plt.show()


def plot_embeddings_mosaic_discrete(list_of_lists,titles,names,y,fsave=None,colors=None,text=None):
    n_rows = len(list_of_lists)
    n_cols = len(list_of_lists[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # make sure axes is 2D
    axes = axes if n_rows > 1 else [axes]
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            X, idx = list_of_lists[i][j]
            y_values = y[idx]
            # scatter plot
            for k,label in enumerate(np.unique(y)):
                mask = label == y_values
                if colors is not None:
                    ax.scatter(X[mask, 0], X[mask, 1], s=5,alpha=0.6,c=colors[k],label=label,edgecolors="none")
                else:
                    ax.scatter(X[mask, 0], X[mask, 1], s=5,alpha=0.6,label=label,edgecolors="none")

            # only left-most plots get the y-axis title
            if j == 0:
                ax.set_ylabel(titles[i])
            
            if text is not None:
                ax.text(0.99,0.01,f"kNN-Accuracy{text[i][j]:.2f}",transform=ax.transAxes,ha="right",va='bottom')
            
            # only top-row plots get the plot title
            if i == 0:
                ax.set_title(names[j])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top","right","left","bottom"]].set_visible(False)

            if i == 0 and j == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    
    fig.legend(legend_handles, legend_labels, loc='center left',markerscale=5, bbox_to_anchor=(1.0, 0.5), title='Classes')

    plt.subplots_adjust(right=0.8)
    
    save(fsave)
    plt.tight_layout()
    plt.show()


def plot_embeddings_row_discrete(list_of_lists,titles,y,fsave=None,colors=None):
    
    n_cols = len(list_of_lists)
    fig, axes = plt.subplots(1, ncols=n_cols, figsize=(4 * n_cols, 4 * 1))
    

    for i in range(n_cols):
        ax = axes[i]
        X, idx = list_of_lists[i]
        y_values = y[idx]
        # scatter plot
        for j,label in enumerate(np.unique(y)):
            mask = label == y_values
            if colors is not None:
                ax.scatter(X[mask, 0], X[mask, 1], s=5,alpha=0.6,c=colors[j],label=label,edgecolors="none")
            else:
                ax.scatter(X[mask, 0], X[mask, 1], s=5,alpha=0.6,label=label,edgecolors="none")


        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top","right","left","bottom"]].set_visible(False)

        if i == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    
    fig.legend(legend_handles, legend_labels, loc='lower center',markerscale=5, bbox_to_anchor=(0.5,-0.02),ncol=len(np.unique(y)),)
    plt.subplots_adjust(bottom=0.8)

    save(fsave)
    plt.tight_layout()
    plt.show()


def plot_embedding_red_box(emb, landmarks, title, y, x1_rects, x2_rects, pos_pictures, fsave=None, colors=None):
    # make sure axes is 2D
    #plg.plot_embedding_red_box(V,landmarks=V[:,0]-1,title="Red Box",y=V[:,0],x1_rects=[(2,4)],x2_rects=[(2,4)])
    y_values = y[landmarks]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    for k, label in enumerate(np.unique(y)):
        mask = label == y_values
        if colors is not None:
            plt.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6, c=[colors[k]], edgecolors="none")
        else:
            plt.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6, edgecolors="none")

    # Add red rectangle
    for x1_rect, x2_rect in zip(x1_rects,x2_rects):
        x1_min, x1_max = x1_rect
        x2_min, x2_max = x2_rect
        rect_width = x1_max - x1_min
        rect_height = x2_max - x2_min
        red_rect = Rectangle((x1_min, x2_min), rect_width, rect_height,
                            linewidth=1.5, edgecolor='black', facecolor='none')
        ax.add_patch(red_rect)

    

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.title(title)

    if fsave:
        plt.savefig(fsave, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return 


def plot_cluster(X,emb,x1_rects,x2_rects,pixel_width,use_mean=True,fsave=None):
    (x1_min,x1_max),(x2_min,x2_max) = x1_rects,x2_rects
    mask = emb[:,0] > x1_min
    mask = mask & emb[:,0] < x1_max
    mask = mask & emb[:,1] > x2_min
    mask = mask & emb[:,1] < x2_max
    data = X[mask,:]
    if use_mean:
        image = data.mean(axis=0).astype(int)
    else:
        image = data.median(axis=0).astype(int)
    image.reshape(pixel_width,-1)
    plt.imshow(image,cmap="gray")
    plt.xticks([])
    plt.yticks([])

    if fsave:
        plt.savefig(fsave, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


def plot_embedding_red_box(emb, landmarks, title, y, x1_rects, x2_rects, pos_pictures, X=None, fsave=None, colors=None,pixel_width=32,channel=1):
    y_values = y[landmarks]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    for k, label in enumerate(np.unique(y)):
        mask = label == y_values
        if colors is not None:
            plt.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6, c=[colors[k]], edgecolors="none")
        else:
            plt.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6, edgecolors="none")

    # Draw red boxes and optionally insert image thumbnails
    for (x1_min, x1_max), (x2_min, x2_max), (px, py,size) in zip(x1_rects, x2_rects, pos_pictures):
        rect_width = x1_max - x1_min
        rect_height = x2_max - x2_min
        red_rect = Rectangle((x1_min, x2_min), rect_width, rect_height,
                             linewidth=1.5, edgecolor='black', facecolor='none')
        ax.add_patch(red_rect)

        if X is not None:
            # mask points in this region
            mask = (emb[:, 0] > x1_min) & (emb[:, 0] < x1_max) & (emb[:, 1] > x2_min) & (emb[:, 1] < x2_max)
            image = plot_cluster_in_func(X, mask,pixel_width,channel)

            if image is not None:
                    ax.imshow(image,cmap="gray" if channel ==1 else None, extent=(px, px + size, py, py + size), aspect='auto', zorder=3)


    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.title(title)
    if fsave:
        plt.savefig(fsave, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


def plot_cluster_in_func(X, mask, pixel_height,channel, use_mean=True):
    if X is None:
        return None
    data = X[mask, :]
    if data.size == 0:
        return None
    if use_mean:
        image = data.mean(axis=0).astype(int)
    else:
        image = np.median(data, axis=0).astype(int)
    if channel ==1:
        image=image.reshape(pixel_height,-1)
    else:
        image=image.reshape(3,pixel_height,-1)
        image= np.transpose(image, (1, 2, 0))
    image = np.clip(0,255,image)
    return image.astype(np.uint8)

from matplotlib.patches import Rectangle

def plot_embedding_red_box1(
    emb, landmarks, title, y,
    x1_rects, x2_rects, pos_pictures,
    X=None, pixel_height=28, use_mean=True,
    channels=1, fsave=None, colors=None, 
    width=20,height=20,x_lim=(-100,100),y_lim=(-100,100),
    show_axis=False
):
    y_values = y[landmarks]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    X_min,X_max,Y_min,Y_max = np.min(emb[:,0]),np.max(emb[:,0]),np.min(emb[:,1]),np.max(emb[:,1])
    for k, label in enumerate(np.unique(y)):
        mask = label == y_values
        color = [colors[k]] if colors is not None else None
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6, c=color, edgecolors="none")
    for (x1_min, x1_max), (x2_min, x2_max), (px, py) in zip(x1_rects, x2_rects, pos_pictures):
        
        # Show image if X is provided
        if X is not None:
            X_here = X[landmarks,:]
            mask = (emb[:, 0] > x1_min) & (emb[:, 0] < x1_max) & \
                   (emb[:, 1] > x2_min) & (emb[:, 1] < x2_max)
            image = plot_cluster_in_func(X_here, mask, pixel_height, channels,use_mean)
            if image is not None:
                if channels==1:
                    ax.imshow(
                    image,
                    extent=(px, px + width, py, py + height),
                    zorder=3,
                    interpolation="bilinear",
                    cmap="grey"
                )
                else:
                    ax.imshow(
                        image,
                        extent=(px, px + width, py, py + height),
                        zorder=3,
                        interpolation="bilinear"
                    )
        red_rect = Rectangle((x1_min, x2_min), x1_max - x1_min, x2_max - x2_min,
                             linewidth=1.5, edgecolor='black', facecolor='none')
        ax.add_patch(red_rect)

        rect_x_min = x1_min
        rect_x_max = x1_max
        rect_y_min = x2_min
        rect_y_max = x2_max

        image_center_x = px + width / 2
        image_center_y = py + height / 2

        rect_edge_x = np.clip(image_center_x, rect_x_min, rect_x_max)
        rect_edge_y = np.clip(image_center_y, rect_y_min, rect_y_max)

        ax.plot(
            [rect_edge_x, image_center_x],
            [rect_edge_y, image_center_y],
            color="black",
            linewidth=1,
            linestyle="--",
            zorder=2
        )
    
    ax.set_ylim(y_lim[0],y_lim[1])
    ax.set_xlim(x_lim[0],x_lim[1])

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.title(title)
    if fsave:
        plt.savefig(fsave, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


#from matplotlib documentation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_radar_from_df_normalized(df, frame='circle', title="Radar Chart of Graph Metrics"):
    labels = df.columns.tolist()
    data = df.values
    theta = radar_factory(len(labels), frame=frame)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='radar'))
    fig.suptitle(title, fontsize=16, y=1.05)

    for i, row in enumerate(data):
        ax.plot(theta, row, label=df.index[i])
        ax.fill(theta, row, alpha=0.1)

    ax.set_varlabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()



def plot_radar_from_df(df, frame='circle', title="Radar Chart with True Value Grid Labels", max_ranges=None):
    labels = df.columns.tolist()
    num_vars = len(labels)
    theta = radar_factory(num_vars, frame=frame)

    # Determine per-axis max (used for scaling)
    if max_ranges is None:
        max_ranges = df.max(axis=0).values
    else:
        assert len(max_ranges) == num_vars

    # Normalize for plotting but retain true values for labels
    df_norm = df.copy()
    for i, col in enumerate(df.columns):
        df_norm[col] = df[col] / max_ranges[i]

    # Grid levels (normalized positions), exclude outermost
    grid_levels = [0.25, 0.5, 0.75,1.0]
    all_grid_levels = grid_levels # used for drawing rings

    # Start plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    fig.suptitle(title, fontsize=16, y=1.1)

    # Plot radar lines
    for idx, row in df_norm.iterrows():
        ax.plot(theta, row.tolist(), label=idx)
        ax.fill(theta, row.tolist(), alpha=0.1)

    # Set uniform radial gridlines
    ax.set_ylim(0, 1)
    ax.set_rgrids(all_grid_levels, labels=['' for _ in all_grid_levels])  # remove default labels

    # Set axis (variable) labels
    #ax.set_varlabels(labels)

    for i in range(num_vars):
        for gv in grid_levels:  # includes 0.0 now
            angle = theta[i]
            value = gv * max_ranges[i]

            # shift slightly *before* the grid line
            radial_offset = -0.05 if gv > 0 else 0

            ax.text(
                angle,
                gv + radial_offset,
                f"{value:.2f}",
                fontsize=8,
                ha='center',
                va='top',
                color='gray',
                rotation=0
            )
        
    label_radius = 1.05  # adjust to control distance from center
    for angle in theta:
        ax.plot([angle, angle], [0, 1], color='gray', lw=0.8, linestyle='-')
    for angle, label in zip(theta, labels):
        ax.text(
            angle,
            label_radius,
            label,
            ha='center',
            va='center',
            fontsize=10,
            rotation=np.degrees(angle),
            rotation_mode='anchor'
        )
    ax.xaxis.set_visible(False)

    # Legend
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def compare_df_with_radars(dfs, titles=None, frame='circle', max_ranges=None, labels=None):
    
    num_charts = len(dfs)
    titles = titles or [f"Radar Chart {i+1}" for i in range(num_charts)]

    col_ref = dfs[0].columns.tolist()
    for i, df in enumerate(dfs):
        assert list(df.columns) == col_ref, f"DataFrame {i} has different columns"

    num_vars = len(col_ref)
    theta = radar_factory(num_vars, frame=frame)

    if max_ranges is None:
        combined_df = pd.concat(dfs)
        max_ranges = combined_df.max(axis=0).values
    else:
        assert len(max_ranges) == num_vars

    fig, axes = plt.subplots(1, num_charts, figsize=(6*num_charts, 6), 
                             subplot_kw=dict(projection='radar'))
    if num_charts == 1:
        axes = [axes]

    for i, (df, ax) in enumerate(zip(dfs, axes)):
        df_norm = df.copy()
        for j, col in enumerate(df.columns):
            df_norm[col] = df[col] / max_ranges[j]

        ax.set_ylim(0, 1)
        grid_levels = [0.25, 0.5, 0.75, 1.0]
        ax.set_rgrids(grid_levels, labels=['' for _ in grid_levels])

        for row_idx, row in enumerate(df_norm.itertuples(index=False)):
            label = labels[row_idx] if labels else df.index[row_idx]
            ax.plot(theta, row, label=label)
            ax.fill(theta, row, alpha=0.1)

        for angle in theta:
            ax.plot([angle, angle], [0, 1], color='gray', lw=0.8, linestyle='-')
        for angle, label in zip(theta, col_ref):
            ax.text(angle, 1.05, label, ha='center', va='center',
                    fontsize=10, rotation=np.degrees(angle), rotation_mode='anchor')

        for j in range(num_vars):
            for gv in grid_levels:
                angle = theta[j]
                value = gv * max_ranges[j]
                ax.text(angle, gv - 0.05, f"{value:.2f}", fontsize=8,
                        ha='center', va='top', color='gray')

        ax.set_title(titles[i], fontsize=14, y=1.1)
        ax.xaxis.set_visible(False)

        if i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    plt.show()
