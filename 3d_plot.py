import plotly.graph_objects as go
from decode import *
from position import transitions, positions

def create_skeleton_traces(position):
    traces = []

    # Define connections between joints
    connections = [
        (Joint.Head, Joint.Neck),
        (Joint.Neck, Joint.LeftShoulder),
        (Joint.Neck, Joint.RightShoulder),
        (Joint.LeftShoulder, Joint.LeftElbow),
        (Joint.LeftElbow, Joint.LeftWrist),
        (Joint.RightShoulder, Joint.RightElbow),
        (Joint.RightElbow, Joint.RightWrist),
        (Joint.Neck, Joint.Core),
        (Joint.Core, Joint.LeftHip),
        (Joint.Core, Joint.RightHip),
        (Joint.LeftHip, Joint.LeftKnee),
        (Joint.LeftKnee, Joint.LeftAnkle),
        (Joint.RightHip, Joint.RightKnee),
        (Joint.RightKnee, Joint.RightAnkle),
    ]

    for player in range(2):
        color = 'red' if player == 0 else 'blue'

        # Create lines for skeleton
        for start, end in connections:
            start_pos = position[PlayerJoint(player, start)]
            end_pos = position[PlayerJoint(player, end)]
            traces.append(go.Scatter3d(
                x=[start_pos[0], end_pos[0]],
                y=[start_pos[1], end_pos[1]],
                z=[start_pos[2], end_pos[2]],
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False
            ))

        # Create markers for joints
        x, y, z = [], [], []
        for joint in Joint:
            pos = position[PlayerJoint(player, joint)]
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])

        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color=color),
            name=f'Player {player}'
        ))

    return traces


def plot_3d_positions(position):
    traces = create_skeleton_traces(position)

    # Create the 3D scatter plot
    fig = go.Figure(data=traces)

    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # This keeps the aspect ratio true to the data
        ),
        title='3D Position Visualization',
        height=800,
        width=800
    )

    # Show the plot
    fig.show()

def main():

    backstep = positions[positions['description'] == 'back step pass']['code'].iloc[0]
    top_free = transitions[transitions['description'] == 'top tries to free leg']['start_position'].iloc[0]
    # Call the function to plot an example position
    #encoded_position = "Kjbf9jYHaX2kJ7eF7XU5aM2TJUdm7mVWbJ2jIWaY0WUlhJZgKlhI1UNPiM1uHeoqVLMdnMT5EhkWWALBjmSCGCiKTOHUh7UzHlhKTLGGhTUxHUg1UBF0hSVFKTk8ZdJHo7UuJMqySdLRaERbIpaFZoIpaKPIGraEWdJkbEQjG2bJW2KSioQgHoiBVVHoogPlFuooSzLwvxQZHTwQUgNwrtP0E5tgTjM7o5TtHoqaVnNqpIUzHZqzWvNipcVJIHpDW6GusfQKKqw4S7MMxeUq"
    encoded_position = top_free
    decoded_position = decode_position(encoded_position)
    plot_3d_positions(decoded_position)

if __name__ == "__main__":
    main()