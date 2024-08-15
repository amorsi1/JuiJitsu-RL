import plotly.graph_objects as go
from plotly.subplots import make_subplots
from position import Joint, Position, calc_limb_distances
import numpy as np


def visualize_positions(pos1, pos2, title1="Position 1", title2="Position 2"):
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(title1, title2)
    )

    # Colors for each player
    colors = ['red', 'blue']

    # Limbs to draw (pairs of joints)
    limbs = [
        (Joint.LeftAnkle, Joint.LeftKnee),
        (Joint.LeftKnee, Joint.LeftHip),
        (Joint.RightAnkle, Joint.RightKnee),
        (Joint.RightKnee, Joint.RightHip),
        (Joint.LeftHip, Joint.Core),
        (Joint.RightHip, Joint.Core),
        (Joint.Core, Joint.Neck),
        (Joint.Neck, Joint.Head),
        (Joint.LeftShoulder, Joint.LeftElbow),
        (Joint.LeftElbow, Joint.LeftWrist),
        (Joint.RightShoulder, Joint.RightElbow),
        (Joint.RightElbow, Joint.RightWrist),
    ]

    def add_position_to_subplot(pos, row, col):
        for player in range(2):
            # Plot joints
            x = [pos[(player, joint.value)][0] for joint in Joint]
            y = [pos[(player, joint.value)][1] for joint in Joint]
            z = [pos[(player, joint.value)][2] for joint in Joint]

            fig.add_trace(
                go.Scatter3d(x=x, y=y, z=z, mode='markers',
                             marker=dict(size=5, color=colors[player]),
                             name=f'Player {player + 1} Joints'),
                row=row, col=col
            )

            # Plot limbs
            for joint1, joint2 in limbs:
                x = [pos[(player, joint1.value)][0], pos[(player, joint2.value)][0]]
                y = [pos[(player, joint1.value)][1], pos[(player, joint2.value)][1]]
                z = [pos[(player, joint1.value)][2], pos[(player, joint2.value)][2]]

                fig.add_trace(
                    go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                 line=dict(color=colors[player], width=5),
                                 name=f'Player {player + 1} {joint1.name}-{joint2.name}'),
                    row=row, col=col
                )

    # Add both positions to the subplots
    add_position_to_subplot(pos1, 1, 1)
    add_position_to_subplot(pos2, 1, 2)

    # Update layout for better viewing
    fig.update_layout(height=800, width=1600, title_text="Position Comparison")
    fig.update_scenes(
        aspectmode='data',
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )

    # Show the plot
    fig.show()

def visualize_3d(list1,list2):
    def create_3d_scatter(points, title):
        x, y, z = np.array(points).T
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,
                colorscale='Viridis',
                opacity=0.8
            ),
            name=title
        )
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
        'pos1', 'pos2')
    )
    fig.add_trace(create_3d_scatter(list1, 'pos 1'), row=1, col=1)
    fig.add_trace(create_3d_scatter(list2, 'pos 2'), row=1, col=2)

    # Update the layout
    fig.update_layout(
        title='3D Scatter Plot of Two Positions',
        height=2000,
        width=2000,
        scene=dict(aspectmode='cube'),
        scene2=dict(aspectmode='cube'),
    )

    # Show the plot
    fig.show()




def visualize_distances(pos1,pos2):
    def create_3d_scatter(points, title):
        x, y, z = np.array(points).T
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,
                colorscale='Viridis',
                opacity=0.8
            ),
            name=title
        )

    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],[{'type': 'scene'}, {'type': 'scene'}],[{'type': 'scene'}, {'type': 'scene'}],[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('pos 1 - player 1', 'from distances', 'pos 1 - player 2','from distances', 'pos 2 - player 1','from distances','pos 2 - player 2''from distances',)
    )

    #calculate coordinates from distances:
    pos1_p1,pos1_p2 = add_distances_to_pos(pos1,calc_limb_distances(pos1))
    pos2_p1, pos2_p2 = add_distances_to_pos(pos2, calc_limb_distances(pos2))

    # Add traces for both positions
    fig.add_trace(create_3d_scatter(list(pos1.player0.values()), 'pos 1 - player 1'), row=1, col=1)
    fig.add_trace(create_3d_scatter(pos1_p1, 'from distances'), row=1, col=2)

    fig.add_trace(create_3d_scatter(list(pos1.player1.values()), 'pos 1 - player 2'), row=2, col=1)
    fig.add_trace(create_3d_scatter(pos1_p2, 'from distances'), row=2, col=2)

    fig.add_trace(create_3d_scatter(list(pos2.player0.values()), 'pos 2 - player 1'), row=3, col=1)
    fig.add_trace(create_3d_scatter(pos2_p1, 'from distances'), row=3, col=2)

    fig.add_trace(create_3d_scatter(list(pos2.player1.values()), 'pos 2 - player 2'), row=4, col=1)
    fig.add_trace(create_3d_scatter(pos2_p2, 'from distances'), row=4, col=2)



    # Update the layout
    fig.update_layout(
        title='3D Scatter Plot of Two Positions',
        height=3000,
        width=3000,
        scene=dict(aspectmode='cube'),
        scene2=dict(aspectmode='cube'),
    )

    # Show the plot
    fig.show()

def add_distances_to_pos(pos: Position,distances: dict[int, list]) -> tuple[list,list] :
    def add_distance_to_player(player_num: int, pos=pos, distances=distances) -> list:
        assert player_num in (0,1), 'player number should be 0 or 1'
        pos_head = pos[(player_num, Joint.Head.value)]
        # players_coords = {k: v for k, v in pos.coords.items() if k[0] == player_num}
        pos_translate = []
        for joint_distance in distances[player_num]:
            pos_translate.append(np.add(pos_head,joint_distance))
        return pos_translate
    return add_distance_to_player(0), add_distance_to_player(1)


