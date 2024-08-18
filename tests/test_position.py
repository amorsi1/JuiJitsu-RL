# from position_v3 import *
from Graph.depracated.position import *
from Graph.depracated.plot_3d import *

# def test_applyRotation():
#     p = Position(positions[positions['description'] == 'completed imanari roll']['code'].iloc[0])
#     t = Position(transitions[transitions['description'] == 'to honey']['start_position'].iloc[1])
#     def normal_rotation(pos):
#         p0h, p1h = pos[(0, Joint.Head.value)], pos[(1, Joint.Head.value)]
#         return -angle(xz(p1h - p0h))
#     def normal_translation(pos):
#         center = np.mean([v for v in pos.coords.values()], axis=0)
#         return -center
#
#     # orient canonically and visualize p
#     # rotation_angle = normal_rotation(p)
#     rotation_angle = 180
#     offset = normal_translation(apply(Reorientation(np.zeros(3), rotation_angle), p))
#     reorientation_obj = Reorientation(offset, rotation_angle)
#     print(offset)
#     reorientation_obj.offset = np.zeros(3,) # to test rotation only
#     rotated = apply(reorientation_obj, p)
#     visualize_positions(p,rotated)

    # orient canonically and visualize t
    # rotation_angle = normal_rotation(t)
    # offset = normal_translation(apply(Reorientation(np.zeros(3), rotation_angle), t))
    # reorientation_obj = Reorientation(offset, rotation_angle)
    # print(offset)
    # # reorientation_obj.offset = np.zeros(3, )  # to test rotation only
    # rotated = apply(reorientation_obj, t)
    # visualize_positions(t, rotated)


# def test_calc_limb_distances():
#     backstep = positions[positions['description'] == 'back step pass']['code'].iloc[0]
#     top_free = transitions[transitions['description'] == 'top tries to free leg']['start_position'].iloc[0]
#     pos1 = Position(backstep)
#     pos2 = Position(top_free)
#     visualize_distances(pos1,pos2)

def test_swap_players():
    honey = Position(transitions[transitions['description'] == 'to honey']['start_position'].iloc[1])
    honey_swapped = swap_players(honey)
    # visualize_positions(honey,honey_swapped)
    swapped = positions_are_equivalent(honey, honey_swapped)
    if swapped is not True:
        raise ValueError('swapped positions should be detected as equivalent')

def test_procrustes_analysis():
    # honey = Position(transitions[transitions['description'] == 'to honey']['start_position'].iloc[1])
    # honey_swapped = swap_players(honey)
    imanari = Position(positions[positions['description'] == 'completed imanari roll']['code'].iloc[0])
    backstep = Position(positions[positions['description'] == 'back step pass']['code'].iloc[0])
    visualize_positions(imanari,backstep)
    mtx1, mtx2, disparity = procrustes(pos_to_list(imanari), pos_to_list(backstep))
    visualize_positions(Position(list(mtx1)),Position(list(mtx2)))
    print(disparity)

def test_imanari_honey():
    # these should be equivalent
    imanari = positions[positions['description'] == 'completed imanari roll']['code'].iloc[0]
    honey = transitions[transitions['description'] == 'to honey']['start_position'].iloc[1]
    same_pos =  positions_are_equivalent(Position(imanari), Position(imanari))
    if same_pos is not True:
        visualize_positions(Position(imanari),Position(imanari))
        raise ValueError("Position should be equivalent to itself")
    diff_pos = positions_are_equivalent(Position(imanari), Position(honey))
    if diff_pos is not True:
        visualize_positions(Position(imanari),Position(honey))
        raise ValueError("Positions should be equivalent")

def test_backstep_topfree():
    imanari = positions[positions['description'] == 'completed imanari roll']['code'].iloc[0]
    backstep = positions[positions['description'] == 'back step pass']['code'].iloc[0]
    top_free = transitions[transitions['description'] == 'top tries to free leg']['start_position'].iloc[0]
    print('imanari v backstep')
    diff_pos = positions_are_equivalent(Position(imanari), Position(backstep))
    if diff_pos is True:
        visualize_positions(Position(imanari),Position(backstep))
        raise ValueError("Completely different positions should not be equivalent")
    print('backstep v top free')
    same_pos = positions_are_equivalent(Position(backstep), Position(top_free))
    if same_pos is not True:
        visualize_positions(Position(backstep), Position(top_free))
        # visualize_positions(orient_canonically_with_mirror(Position(backstep)), orient_canonically_with_mirror(Position(top_free)))
        # if positions_are_equivalent(orient_canonically_with_mirror(Position(backstep)), orient_canonically_with_mirror(Position(top_free))) is not True:
        raise ValueError("Positions should be equivalent")

