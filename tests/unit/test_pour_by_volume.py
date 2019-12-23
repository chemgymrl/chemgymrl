from chemistrylab import *
from numpy.testing import assert_almost_equal

'''
tests list:
Pouring:
test_pouring_without_overflow(): Pour from A to B, without overflow
1.  input:
        A is empty; B has [H2O, C6H14, Na+, Cl-], volume = v_max/2; poured_volume = 0;
    assertion:
        the material_dict and solute_dict in A is empty before and after pouring;
        the material_dict and solute_dict in B is unchanged (length, keys, instance, values);
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air);
2.  input:
        A is empty; B has [H2O, C6H14, Na+, Cl-], volume = v_max/2; poured_volume = v_max;
    assertion:
        the material_dict and solute_dict in A is empty before and after pouring;
        the material_dict and solute_dict in B is unchanged (length, keys, values);
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air);
3.  input:
        A has [H2O, C6H14, Na+, Cl-], volume = v_max; B is empty; poured_volume = 0;
    assertion:
        A's material_dict and solute_dict is unchanged (length, keys, instance, values);
        the material_dict and solute_dict in B is empty before and after pouring;
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air);
4.  input:
        A has [H2O, C6H14, Na+, Cl-], volume = v_max; B is empty; poured_volume = v_max;
    assertion:
        the material_dict and solute_dict in A is empty after pouring;
        B should have identical solute_dict and material_dict as A used to have (length, keys, instance, values);
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air);
5.  input:
        A has [H2O, C6H14, Na+, Cl-], volume = v_max; B has [H2O, C6H14, Na+, Cl-], volume = v_max/4; poured_volume = v_max/2
    assertion:
        in the material_dict, sum of each material in both A and B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in both A and B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)
6.  input:
        A has [H2O, C6H14, Na+, Cl-], volume = v_max/2; B has [H2O, C6H14, Na+, Cl-], volume = v_max/4; poured_volume = v_max/2
    assertion:
        A should be empty
        in the material_dict, sum of each material in B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in both B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)
7.  input:
        A has [H2O, Na+, Cl-], volume = v_max; B has [C6H14], volume = v_max/4; poured_volume = v_max/2
    assertion:
        in the material_dict, sum of each material in both A and B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in both A and B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)
8.  input:
        A has [H2O, Na+, Cl-], volume = v_max/2; B has [C6H14], volume = v_max/4; poured_volume = v_max/2
    assertion:
        A should be empty
        in the material_dict, sum of each material in B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)
9.  input:
        A has [C6H14], volume = v_max; B has [H2O, Na+, Cl-], volume = v_max/4; poured_volume = v_max/2
    assertion:
        in the material_dict, sum of each material in both A and B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in both A and B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)
10.  input:
        A has [C6H14], volume = v_max/2; B has [H2O, Na+, Cl-], volume = v_max/4; poured_volume = v_max/2
    assertion:
        A should be empty
        in the material_dict, sum of each material in B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)


test_pouring_with_overflow(): Pour from A to B, with overflow
1.  input:
        A has [H2O, C6H14, Na+, Cl-], volume = v_max; B has [H2O, C6H14, Na+, Cl-], volume = v_max/4; poured_volume = v_max
    assertion:
        A should be empty
        in the material_dict, sum of each material in both A and B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in both A and B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)
2.  input:
        A has [H2O, Na+, Cl-], volume = v_max; B has [C6H14], volume = v_max/4; poured_volume = v_max
    assertion:
        A should be empty
        B is full (total volume = v_max)
        in the material_dict, sum of each material in both A and B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in both A and B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)
3.  input:
        A has [C6H14], volume = v_max; B has [H2O, Na+, Cl-], volume = v_max/4; poured_volume = v_max
    assertion:
        A should be empty
        B is full (total volume = v_max)
        in the material_dict, sum of each material in both A and B should remain unchanged after pouring;
        in the solute_dict, sum of each solute in both A and B should remain unchanged and be the same as material_dict
        layers_position_dicts for both vessels match the material_dicts (solid + solvent + air)

'''


class Tests:  # H2O, C6H14, Na+, Cl-
    def test_pouring_without_overflow_1(self,
                                        volume=0.0):
        # initialized vessels
        vessel_a = initialize_vessel_empty(label='A',
                                           v_max=1000.0)
        vessel_b = initialize_vessel_h2o_c6h14_na_cl(label='B',
                                                     v_max=1000.0,
                                                     h2o_volume=250.0,
                                                     c6h14_volume=250.0,
                                                     )
        # get material_dict, solute_dict, layers_position_dict
        vessel_a_material_dict = vessel_a.get_material_dict()
        vessel_a_solute_dict = vessel_a.get_solute_dict()
        vessel_b_material_dict = vessel_b.get_material_dict()
        vessel_b_solute_dict = vessel_b.get_solute_dict()

        # pour form A to B
        event = ['pour by volume', vessel_b, volume]
        vessel_a.push_event_to_queue(events=[event])

        # get new material_dict, solute_dict, layers_position_dict
        new_vessel_a_material_dict = vessel_a.get_material_dict()
        new_vessel_a_solute_dict = vessel_a.get_solute_dict()
        new_vessel_b_material_dict = vessel_b.get_material_dict()
        new_vessel_b_solute_dict = vessel_b.get_solute_dict()

        # assertions
        # A's material_dict
        assert (not vessel_a_material_dict)  # empty before pouring
        assert (not new_vessel_a_material_dict)  # empty after pouring
        # A's solute_dict
        assert (not vessel_a_solute_dict)  # empty before pouring
        assert (not new_vessel_a_solute_dict)  # empty after pouring
        # B's material_dict
        assert len(new_vessel_b_material_dict) == len(vessel_b_material_dict)  # length
        for M in new_vessel_b_material_dict:
            assert M == vessel_b_material_dict[M][0].get_name()  # instance
            assert new_vessel_b_material_dict[M][1] == vessel_b_material_dict[M][1]  # key, value
        # B's solute_dict
        assert len(new_vessel_b_solute_dict) == len(vessel_b_solute_dict)  # length
        for Solute in new_vessel_b_solute_dict:
            for Solvent in new_vessel_b_solute_dict[Solute]:
                assert new_vessel_b_solute_dict[Solute][Solvent] == vessel_b_solute_dict[Solute][Solvent]  # key, value
        # A's layers_position_dict
        vessel_a_position_dict, vessel_a_variance = vessel_a.get_position_and_variance()
        for M in new_vessel_a_material_dict:
            if new_vessel_a_material_dict[M][0].is_solute():
                assert M not in vessel_a_position_dict
            else:
                assert M in vessel_a_position_dict
        assert 'Air' in vessel_a_position_dict
        # B's layers_position_dict
        vessel_b_position_dict, vessel_b_variance = vessel_b.get_position_and_variance()
        assert vessel_b_variance != 2.0  # variance should not be reset
        for M in new_vessel_b_material_dict:
            if new_vessel_b_material_dict[M][0].is_solute():
                assert M not in vessel_b_position_dict
            else:
                assert M in vessel_b_position_dict
                assert vessel_b_position_dict[M] != 0.0  # position should not be reset
        assert 'Air' in vessel_b_position_dict

    def test_pouring_without_overflow_2(self,
                                        volume=1000.0):
        # initialized vessels
        vessel_a = initialize_vessel_empty(label='A',
                                           v_max=1000.0)
        vessel_b = initialize_vessel_h2o_c6h14_na_cl(label='B',
                                                     v_max=1000.0,
                                                     h2o_volume=250.0,
                                                     c6h14_volume=250.0,
                                                     )
        # get material_dict, solute_dict, layers_position_dict
        vessel_a_material_dict = vessel_a.get_material_dict()
        vessel_a_solute_dict = vessel_a.get_solute_dict()
        vessel_b_material_dict = vessel_b.get_material_dict()
        vessel_b_solute_dict = vessel_b.get_solute_dict()

        # pour form A to B
        event = ['pour by volume', vessel_b, volume]
        vessel_a.push_event_to_queue(events=[event])

        # get new material_dict, solute_dict, layers_position_dict
        new_vessel_a_material_dict = vessel_a.get_material_dict()
        new_vessel_a_solute_dict = vessel_a.get_solute_dict()
        new_vessel_b_material_dict = vessel_b.get_material_dict()
        new_vessel_b_solute_dict = vessel_b.get_solute_dict()

        # assertions
        # A's material_dict
        assert (not vessel_a_material_dict)
        assert (not new_vessel_a_material_dict)
        # A's solute_dict
        assert (not vessel_a_solute_dict)
        assert (not new_vessel_a_solute_dict)
        # B's material_dict
        assert len(new_vessel_b_material_dict) == len(vessel_b_material_dict)  # length
        for M in new_vessel_b_material_dict:
            assert M == vessel_b_material_dict[M][0].get_name()  # instance
            assert new_vessel_b_material_dict[M][1] == vessel_b_material_dict[M][1]  # key, value
        # B's solute_dict
        assert len(new_vessel_b_solute_dict) == len(vessel_b_solute_dict)  # length
        for Solute in new_vessel_b_solute_dict:
            for Solvent in new_vessel_b_solute_dict[Solute]:
                assert new_vessel_b_solute_dict[Solute][Solvent] == vessel_b_solute_dict[Solute][Solvent]  # key, value
        # A's layers_position_dict
        vessel_a_position_dict, _ = vessel_a.get_position_and_variance()
        for M in new_vessel_a_material_dict:
            if new_vessel_a_material_dict[M][0].is_solute():
                assert M not in vessel_a_position_dict
            else:
                assert M in vessel_a_position_dict
        assert 'Air' in vessel_a_position_dict
        # B's layers_position_dict
        vessel_b_position_dict, vessel_b_variance = vessel_b.get_position_and_variance()
        assert vessel_b_variance != 2.0  # variance should not be reset
        for M in new_vessel_b_material_dict:
            if new_vessel_b_material_dict[M][0].is_solute():
                assert M not in vessel_b_position_dict
            else:
                assert M in vessel_b_position_dict
                assert vessel_b_position_dict[M] != 0.0  # position should not be reset
        assert 'Air' in vessel_b_position_dict

    def test_pouring_without_overflow_3(self,
                                        volume=0.0):
        # initialized vessels
        vessel_a = initialize_vessel_h2o_c6h14_na_cl(label='A',
                                                     v_max=1000.0,
                                                     h2o_volume=500.0,
                                                     c6h14_volume=500.0,
                                                     )
        vessel_b = initialize_vessel_empty(label='B',
                                           v_max=1000.0)

        # get material_dict, solute_dict, layers_position_dict
        vessel_a_material_dict = vessel_a.get_material_dict()
        vessel_a_solute_dict = vessel_a.get_solute_dict()
        vessel_b_material_dict = vessel_b.get_material_dict()
        vessel_b_solute_dict = vessel_b.get_solute_dict()

        # pour form A to B
        event = ['pour by volume', vessel_b, volume]
        vessel_a.push_event_to_queue(events=[event])

        # get new material_dict, solute_dict, layers_position_dict
        new_vessel_a_material_dict = vessel_a.get_material_dict()
        new_vessel_a_solute_dict = vessel_a.get_solute_dict()
        new_vessel_b_material_dict = vessel_b.get_material_dict()
        new_vessel_b_solute_dict = vessel_b.get_solute_dict()

        # assertions
        # A's material_dict
        assert len(new_vessel_a_material_dict) == len(vessel_a_material_dict)  # length
        for M in new_vessel_a_material_dict:
            assert M == vessel_a_material_dict[M][0].get_name()  # instance
            assert new_vessel_a_material_dict[M][1] == vessel_a_material_dict[M][1]  # key, value
        # A's solute_dict
        assert len(new_vessel_a_solute_dict) == len(vessel_a_solute_dict)  # length
        for Solute in new_vessel_a_solute_dict:
            for Solvent in new_vessel_a_solute_dict[Solute]:
                assert new_vessel_a_solute_dict[Solute][Solvent] == vessel_a_solute_dict[Solute][Solvent]  # key, value
        # B's material_dict
        assert (not vessel_b_material_dict)
        assert (not new_vessel_b_material_dict)
        # B's solute_dict
        assert (not vessel_b_solute_dict)
        assert (not new_vessel_b_solute_dict)
        # A's layers_position_dict
        vessel_a_position_dict, _ = vessel_a.get_position_and_variance()
        for M in new_vessel_a_material_dict:
            if new_vessel_a_material_dict[M][0].is_solute():
                assert M not in vessel_a_position_dict
            else:
                assert M in vessel_a_position_dict
        assert 'Air' in vessel_a_position_dict
        # B's layers_position_dict
        vessel_b_position_dict, vessel_b_variance = vessel_b.get_position_and_variance()
        assert vessel_b_variance != 2.0  # variance should not be reset
        for M in new_vessel_b_material_dict:
            if new_vessel_b_material_dict[M][0].is_solute():
                assert M not in vessel_b_position_dict
            else:
                assert M in vessel_b_position_dict
                assert vessel_b_position_dict[M] != 0.0  # position should not be reset
            assert 'Air' in vessel_b_position_dict

    def test_pouring_without_overflow_4(self,
                                        volume=1000.0):
        # initialized vessels
        vessel_a = initialize_vessel_h2o_c6h14_na_cl(label='A',
                                                     v_max=1000.0,
                                                     h2o_volume=500.0,
                                                     c6h14_volume=500.0,
                                                     )
        vessel_b = initialize_vessel_empty(label='B',
                                           v_max=1000.0)

        # get material_dict, solute_dict, layers_position_dict
        vessel_a_material_dict = vessel_a.get_material_dict()
        vessel_a_solute_dict = vessel_a.get_solute_dict()
        vessel_b_material_dict = vessel_b.get_material_dict()
        vessel_b_solute_dict = vessel_b.get_solute_dict()

        # pour form A to B
        event = ['pour by volume', vessel_b, volume]
        vessel_a.push_event_to_queue(events=[event])

        # get new material_dict, solute_dict, layers_position_dict
        new_vessel_a_material_dict = vessel_a.get_material_dict()
        new_vessel_a_solute_dict = vessel_a.get_solute_dict()
        new_vessel_b_material_dict = vessel_b.get_material_dict()
        new_vessel_b_solute_dict = vessel_b.get_solute_dict()

        # assertions
        # A's material_dict
        assert (not new_vessel_a_material_dict)
        # A's solute_dict
        assert (not new_vessel_a_solute_dict)
        # B's material_dict
        assert (not vessel_b_material_dict)
        assert len(new_vessel_b_material_dict) == len(vessel_a_material_dict)  # length
        for M in new_vessel_b_material_dict:
            assert M == vessel_a_material_dict[M][0].get_name()  # instance
            assert new_vessel_b_material_dict[M][1] == vessel_a_material_dict[M][1]  # key, value
        # B's solute_dict
        assert (not vessel_b_solute_dict)
        assert len(new_vessel_b_solute_dict) == len(vessel_a_solute_dict)  # length
        for Solute in new_vessel_b_solute_dict:
            for Solvent in new_vessel_b_solute_dict[Solute]:
                assert new_vessel_b_solute_dict[Solute][Solvent] == vessel_a_solute_dict[Solute][
                    Solvent]  # key, value
        # A's layers_position_dict
        vessel_a_position_dict, _ = vessel_a.get_position_and_variance()
        for M in new_vessel_a_material_dict:
            if new_vessel_a_material_dict[M][0].is_solute():
                assert M not in vessel_a_position_dict
            else:
                assert M in vessel_a_position_dict
        assert 'Air' in vessel_a_position_dict
        # B's layers_position_dict
        vessel_b_position_dict, vessel_b_variance = vessel_b.get_position_and_variance()
        assert vessel_b_variance == 2.0  # variance should be reset
        for M in new_vessel_b_material_dict:
            if new_vessel_b_material_dict[M][0].is_solute():
                assert M not in vessel_b_position_dict
            else:
                assert M in vessel_b_position_dict
                assert vessel_b_position_dict[M] == 0.0  # position should be reset
        assert 'Air' in vessel_b_position_dict


def initialize_vessel_h2o_c6h14_na_cl(label=None,
                                      v_max=1000.0,
                                      h2o_volume=500.0,
                                      c6h14_volume=500.0,
                                      ):
    new_vessel = vessel.Vessel(label=label,
                               v_max=v_max,
                               )

    # initialize materials
    H2O = material.H2O()
    C6H14 = material.C6H14()
    Na = material.Na()
    Cl = material.Cl()
    Na.set_charge(1.0)
    Na.set_solute_flag(True)
    Cl.set_charge(-1.0)
    Cl.set_solute_flag(True)

    # convert volume to mole
    h2o_amount = util.convert_volume_to_mole(volume=h2o_volume,
                                             density=H2O.get_density(),
                                             molar_mass=H2O.get_molar_mass(),
                                             )
    c6h14_amount = util.convert_volume_to_mole(volume=c6h14_volume,
                                               density=C6H14.get_density(),
                                               molar_mass=C6H14.get_molar_mass(),
                                               )

    # material_dict
    material_dict = {H2O.get_name(): [H2O, h2o_amount],
                     C6H14.get_name(): [C6H14, c6h14_amount],
                     Na.get_name(): [Na, 1.0],
                     Cl.get_name(): [Cl, 1.0]
                     }
    solute_dict = {Na.get_name(): {H2O.get_name(): 0.5,
                                   C6H14.get_name(): 0.5
                                   },
                   Cl.get_name(): {H2O.get_name(): 0.5,
                                   C6H14.get_name(): 0.5
                                   },
                   }

    material_dict, solute_dict = util.check_overflow(material_dict=material_dict,
                                                     solute_dict=solute_dict,
                                                     v_max=new_vessel.get_max_volume(),
                                                     )

    event_1 = ['update material dict', material_dict]
    event_2 = ['update solute dict', solute_dict]
    event_3 = ['fully mix']

    new_vessel.push_event_to_queue(events=None, feedback=[event_1, event_2, event_3], dt=0)
    return new_vessel


def initialize_vessel_h2o_na_cl(label=None,
                                v_max=1000.0,
                                h2o_volume=500.0,
                                ):
    new_vessel = vessel.Vessel(label=label,
                               v_max=v_max,
                               )

    # initialize materials
    H2O = material.H2O()
    Na = material.Na()
    Cl = material.Cl()
    Na.set_charge(1.0)
    Na.set_solute_flag(True)
    Cl.set_charge(-1.0)
    Cl.set_solute_flag(True)

    # convert volume to mole
    h2o_amount = util.convert_volume_to_mole(volume=h2o_volume,
                                             density=H2O.get_density(),
                                             molar_mass=H2O.get_molar_mass(),
                                             )

    # material_dict
    material_dict = {H2O.get_name(): [H2O, h2o_amount],
                     Na.get_name(): [Na, 1.0],
                     Cl.get_name(): [Cl, 1.0]
                     }
    solute_dict = {Na.get_name(): {H2O.get_name(): 1.0,
                                   },
                   Cl.get_name(): {H2O.get_name(): 1.0,
                                   },
                   }

    material_dict, solute_dict = util.check_overflow(material_dict=material_dict,
                                                     solute_dict=solute_dict,
                                                     v_max=new_vessel.get_max_volume(),
                                                     )

    event_1 = ['update material dict', material_dict]
    event_2 = ['update solute dict', solute_dict]
    event_3 = ['fully mix']

    new_vessel.push_event_to_queue(events=None, feedback=[event_1, event_2, event_3], dt=0)
    return new_vessel


def initialize_vessel_c6h14(label=None,
                            v_max=1000.0,
                            c6h14_volume=500.0,
                            ):
    new_vessel = vessel.Vessel(label=label,
                               v_max=v_max,
                               )

    # initialize materials
    C6H14 = material.C6H14()

    # convert volume to mole
    c6h14_amount = util.convert_volume_to_mole(volume=c6h14_volume,
                                               density=C6H14.get_density(),
                                               molar_mass=C6H14.get_molar_mass(),
                                               )

    # material_dict
    material_dict = {C6H14.get_name(): [C6H14, c6h14_amount],
                     }

    material_dict, solute_dict = util.check_overflow(material_dict=material_dict,
                                                     solute_dict={},
                                                     v_max=new_vessel.get_max_volume(),
                                                     )

    event_1 = ['update material dict', material_dict]
    event_2 = ['fully mix']

    new_vessel.push_event_to_queue(events=None, feedback=[event_1, event_2], dt=0)
    return new_vessel


def initialize_vessel_empty(label=None,
                            v_max=1000.0,
                            ):
    new_vessel = vessel.Vessel(label=label,
                               v_max=v_max,
                               )
    return new_vessel
