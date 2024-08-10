from position import *
def test_positions_are_equivalent():

    # these should be equivalent
    imanari = positions[positions['description'] == 'completed imanari roll']['code'].iloc[0]
    honey = transitions[transitions['description'] == 'to honey']['start_position'].iloc[1]
    assert positions_are_equivalent(Position(imanari), Position(imanari)) is True, "Should be equivalent to self"
    assert positions_are_equivalent(Position(imanari), Position(honey)) is True, "Aligning start position to node"

    backstep = positions[positions['description'] == 'back step pass']['code'].iloc[0]
    top_free = transitions[transitions['description'] == 'top tries to free leg']['start_position'].iloc[0]
    assert positions_are_equivalent(Position(imanari), Position(backstep)) is not True, "Completely different positions should not be equivalent"
    assert positions_are_equivalent(Position(backstep), Position(top_free)) is True, "Aligning start position to node"


