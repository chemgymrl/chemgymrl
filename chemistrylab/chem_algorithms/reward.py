'''
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.

Module to define a universal reward function.

:title: reward.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-11-22

The reward generated by completing a process in any of the reaction, extraction, or distillation
bench relates to the distribution and purity of the desired material inside the vessel, or one
of the vessels, outputted upon completing all actions required by the bench.

Reaction bench

Reaction bench outputs a single vessel that includes a variety of materials. Therefore, the reward
is determined by the amount of the desired material in relation to the total amount of material.

Extraction bench

The extraction bench outputs 3 vessels each containing materials, however, not every vessel may
contain the desired material. Therefore, the reward is determined by the purity of the desired
material in each vessel and the number of materials that contain a non-negligible amount of the
desired material. A greater purity of the desired material in a vessel will yield a greater reward
as will the desired material being present in fewer vessels. However, a result that gives a greater
purity of the desired material will be more rewarded than a result that has the desired material
confined to fewer vessels.

Distillation bench
'''

import copy
import math
import numpy as np
import sys

sys.path.append("../../")
from chemistrylab.chem_algorithms import material

# ---------- # REACTION BENCH # ---------- #

class ReactionReward:
    '''
    Class object to define a reward function for the Reaction Bench environment.

    Parameters
    ---------------
    `vessel` : `vessel` (default=`None`)
        A vessel object containing state variables, materials, solutes, and spectral data.
    `desired_material` : `str` (default=`""`)
        The name of the required output material that has been designated as reward.

    Returns
    ---------------
    None

    Raises
    ---------------
    None
    '''

    def __init__(
        self,
        vessel=None,
        desired_material=""
    ):
        '''
        Constructor class for `ReactionReward` environment.
        '''

        self.vessel, self.desired_material = self._check_parameters(
            vessel=vessel,
            desired_material=desired_material
        )

    def _check_parameters(self, vessel=None, desired_material=""):
        '''
        Method to validate the inputted parameters. The desired material must be a string object
        and present in the vessel's `material_dict`. The vessel must be a vessel object and
        contain at least one material.

        Parameters
        ---------------
        `vessel` : `vessel` (default=`None`)
            A vessel object containing state variables, materials, solutes, and spectral data.
        `desired_material` : `str` (default=`None`)
            The name of the required output material that has been designated as reward.

        Returns
        ---------------
        `vessel` : `vessel` (default=`None`)
            A vessel object containing state variables, materials, solutes, and spectral data.
        `desired_material` : `str` (default=`None`)
            The name of the required output material that has been designated as reward.

        Raises
        ---------------
        None
        '''

        # ensure the desired material is represented by a string
        if not isinstance(desired_material, str):
            print("Invalid Parameter Type: `desired_material` must be a string.")
            desired_material = ""

        # acquire all the materials from the inputted vessel
        all_materials = [material for material, __ in vessel._material_dict.items()]

        # check that the input vessel has at least one material
        if not all_materials:
            print("Vessel has no materials.")

        # check that the inputted vessel has the desired material
        if any([
                not desired_material,
                desired_material not in all_materials
        ]):
            print("Desired material not found in input vessel.")

        return vessel, desired_material

    def calc_reward(self):
        '''
        Method to calculate the reward accumulated from completing the Reaction Bench. The reward
        is calculated using the purity of the desired material in the outputted vessel object. The
        ratio of desired material to total material gives the purity reward. Therefore, the reward
        value must range between 0 and 1.
        '''

        # create variables to contain the material amounts
        desired_material_amount = 0
        total_material_amount = 0

        # unpack the vessel's material dictionary
        for material, value_list in self.vessel._material_dict.items():
            # unpack the value list
            material_amount = value_list[1]

            # add the amount of material to the list of total material
            total_material_amount += material_amount

            # check if the current material is the desired material;
            # if so, set the amount of desired material produced
            if material == self.desired_material:
                desired_material_amount = material_amount

        # calculate the reward
        reward = desired_material_amount/total_material_amount

        return reward

# ---------- # EXTRACTION BENCH # ---------- #

class ExtractionReward:
    '''
    Class object to define a reward function for the Extraction Bench environment.

    Parameters
    ---------------
    `vessels` : `list` (default=`[]`)
        A list of vessel objects outputted by the Extraction Bench environment.
    `desired_material` : `str` (default=`""`)
        The name of the required output material that has been designated as reward.
    `initial_target_material` : `float` (default=`0.0`)
        The initial amount of target material in the vessel first provided to the Extraction Bench
        environment.

    Returns
    ---------------
    None

    Raises
    ---------------
    None
    '''

    def __init__(
            self,
            vessels=[],
            desired_material="",
            initial_target_amount=0.0
    ):
        '''
        Constructor class for the `ExtractionReward` environment.
        '''

        self.vessels, self.desired_material, self.desired_vessels = self._check_parameters(
            vessels=vessels,
            desired_material=desired_material
        )

        self.initial_target_amount = initial_target_amount

    def _check_parameters(self, vessels, desired_material):
        '''
        Method to validate the inputted parameters. The desired material must be a string object
        and present in the vessel's `material_dict`. The vessel must be a vessel object and
        contain at least one material.

        Parameters
        ---------------
        `vessel` : `vessel` (default=`None`)
            A vessel object containing state variables, materials, solutes, and spectral data.
        `desired_material` : `str` (default=`None`)
            The name of the required output material that has been designated as reward.

        Returns
        ---------------
        `vessel` : `vessel` (default=`None`)
            A vessel object containing state variables, materials, solutes, and spectral data.
        `desired_material` : `str` (default=`None`)
            The name of the required output material that has been designated as reward.

        Raises
        ---------------
        None
        '''

        # ensure the desired material is represented by a string
        if not isinstance(desired_material, str):
            print("Invalid Parameter Type: `desired_material` must be a string.")
            desired_material = ""

        # create a list to track which vessels contain the desired material
        is_present = [True for __ in vessels]

        # check that the desired material is in at least one of the inputted vessels
        for i, vessel in enumerate(vessels):
            # get the name of the vessel
            label = vessel.label

            # acquire all the materials from the inputted vessel
            all_materials = [material for material, __ in vessel._material_dict.items()]

            # check that the input vessel has at least one material
            if not all_materials:
                print("{} has no materials.".format(label))

            # check that the inputted vessel has the desired material;
            # if it does not change the ith element in `is_present` to False
            if any([
                    not desired_material,
                    desired_material not in all_materials
            ]):
                is_present[i] = False
                print("Desired material not found in {}.".format(label))

        # if no vessel contains the desired material, all elements in `is_present` will be False
        if not any(is_present):
            print("No desired material found in any inputted vessels.")

        # create a new list of vessels that contain the desired material
        desired_vessels = [vessels[i] for i, desired in enumerate(is_present) if desired]

        return vessels, desired_material, desired_vessels

    @staticmethod
    def calc_vessel_purity(vessel, desired_material, initial_target_amount):
        '''
        Method to calculate the full reward once the final action has taken place.

        Parameters
        ---------------
        `vessel` : `vessel.Vessel`
            A vessel object that contains all of the extracted materials and solutes.
        `desired_material` : `str`
            The name of the required output material that has been designated as reward.
        `initial_target_amount` : `float`
            The initial amount of target material in the vessel first provided to the Extraction Bench
            environment.

        Returns
        ---------------
        `reward` : `float`
            The purity of the desired material with respect to the total desired material available in
            the original vessel provided by the Reaction Bench.

        Raises
        ---------------
        `AssertionError`:
            Raised if no target material is found in the extraction vessel.
        '''

        # acquire the amount of desired material from the `vessel` parameter
        material_amount = vessel.get_material_amount(desired_material)

        # set a negative reward if no desired material was made available to the Extraction Bench
        if abs(material_amount - 0) < 1e-6:
            reward = 0.0
        else:
            try:
                # ensure the initial, available desired material is non-negligible
                assert abs(initial_target_amount - 0.0) > 1e-6

                # find the ratio of the current desired material to the available desired material
                reward = material_amount / initial_target_amount

                print(
                    "done_reward ({}): {}, in_vessel: {}, initial: {}".format(
                        vessel.label,
                        "{} %".format(round(reward, 2)),
                        "{:e}".format(material_amount),
                        "{:e}".format(initial_target_amount)
                    )
                )

            except AssertionError:
                reward = 0
                print(
                    "Error: There was no target material in the original vessel passed to the "
                    "Extraction Bench."
                )

        return reward

    def calc_reward(self):
        '''
        Method to calculate the reward accumulated from completing the Extraction Bench. Upon
        completing the Extraction Bench, several vessels are outputted each containing materials
        that may or may not include the desired material. No material is lost, so the amount of
        target material in the initial extraction vessel is spead about the vessels upon completing
        the Extraction Bench. Therefore, a greater reward should be given to a greater purity and
        the desired material being present in fewer separate vessels.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `reward` : `float`
            The final reward calculated from all vessels outputted by the Extraction Bench.

        Raises
        ---------------
        None
        '''

        total_reward = 0

        # calculate the purity reward in each vessel (ranges between 0 and 1)
        for vessel in self.desired_vessels:
            total_reward += self.calc_vessel_purity(
                vessel=vessel,
                desired_material=self.desired_material,
                initial_target_amount=self.initial_target_amount
            )

        # calculate the final reward by dividing the sum of the purity rewards by the number of
        # vessels containing at least some of the desired material
        if len(self.desired_vessels) == 0:
            final_reward = 0
        else:
            final_reward = total_reward / len(self.desired_vessels)

        return final_reward

    def validate_vessels(self, purity_threshold=0.0):
        '''
        Method to examine each vessel outputted by the Extraction Bench and determine if it passed
        the purity threshold and should be outputted/saved.

        Parameters
        ---------------
        `purity_threshold` : `float` (default=`0.0`)
            The minimum purity a vessel must exceed to be saved and used in future processes.

        Returns
        ---------------
        `validated_vessels` : `list` (default=`[]`)
            A list of vessel objects that each contain enough of the desired material such that the
            purity of the desired material exceeds the `purity_threshold` parameter.

        Raises
        ---------------
        None
        '''

        # enusre the purity threshold is not negative and does not exceed 1.0;
        # if so, use the default purity threshold instead
        if any([
                purity_threshold < 0.0,
                purity_threshold > 1.0
        ]):
            purity_threshold = 0.0

        # create a space to store vessels with purity that exceeds the threshold
        validated_vessels = []

        # iterate through each vessel that contains at least some of the desired material
        for vessel in self.desired_vessels:

            # determine the purity reward of each vessel
            reward = self.calc_vessel_purity(
                vessel=vessel,
                desired_material=self.desired_material,
                initial_target_amount=self.initial_target_amount
            )

            # check if the purity reward exceeds the defined threshold
            if reward > purity_threshold:
                validated_vessels.append(vessel)

        return validated_vessels

# ---------- # DISTILLATION BENCH # ---------- #

class DistillationReward:
    '''
    Class object to define a reward function for the Distillation Bench environment.

    Parameters
    ---------------
    `vessels` : `list` (default=`[]`)
        A list of vessel objects outputted by the Distillation Bench environment.
    `desired_material` : `str` (default=`""`)
        The name of the required output material that has been designated as reward.

    Returns
    ---------------
    None

    Raises
    ---------------
    None
    '''

    def __init__(
            self,
            vessels=[],
            desired_material=""
    ):
        '''
        Constructor class for the `DistillationReward` environment.
        '''

        # validate the inputted parameters
        self.vessels, self.desired_material, self.desired_vessels = self._check_parameters(
            vessels=vessels,
            desired_material=desired_material
        )

    def _check_parameters(self, vessels, desired_material):
        '''
        Method to validate the parameters defined by the constructor class.

        Parameters
        ---------------
        `vessels` : `list` (default=`[]`)
            A list of vessel objects outputted by the Distillation Bench environment.
        `desired_material` : `str` (default=`""`)
            The name of the required output material that has been designated as reward.

        Returns
        ---------------
        `vessels` : `list` (default=`[]`)
            A list of vessel objects outputted by the Distillation Bench environment.
        `desired_material` : `str` (default=`""`)
            The name of the required output material that has been designated as reward.
        `desired_vessels` : `list` (default=`[]`)
            A list of vessel objects outputted by the Distillation Bench environment that include
            the desired material.

        Raises
        ---------------
        None
        '''

        # ensure the desired material is represented by a string
        if not isinstance(desired_material, str):
            print("Invalid Parameter Type: `desired_material` must be a string.")
            desired_material = ""

        # create a list to track which vessels contain the desired material
        is_present = [True for __ in vessels]

        # check that the desired material is in at least one of the inputted vessels
        for i, vessel in enumerate(vessels):
            # get the name of the vessel
            label = vessel.label

            # acquire all the materials from the inputted vessel
            all_materials = [material for material, __ in vessel._material_dict.items()]

            # check that the input vessel has at least one material;
            # if it does not change the ith element in `is_present` to False
            if not all_materials:
                is_present[i] = False
                print("{} has no materials.".format(label))
                

            # check that the inputted vessel has the desired material;
            # if it does not change the ith element in `is_present` to False
            if any([
                    not desired_material,
                    desired_material not in all_materials
            ]):
                is_present[i] = False
                print("Desired material not found in {}.".format(label))

        # if no vessel contains the desired material, all elements in `is_present` will be False
        if not any(is_present):
            print("No desired material found in any inputted vessels.")

        # create a new list of vessels that contain the desired material
        desired_vessels = [vessels[i] for i, desired in enumerate(is_present) if desired]

        return vessels, desired_material, desired_vessels

    @staticmethod
    def calc_vessel_purity(vessel, desired_material):
        '''
        Method to calculate the purity reward of the current vessel with respect to the other
        materials in the vessel.

        Parameters
        ---------------
        `beaker` : `vessel.Vessel`
            A vessel object that contains all of the extracted materials and solutes.
        `desired_material` : `str`
            The name of the required output material that has been designated as reward.

        Returns
        ---------------
        `reward` : `float`
            The purity ratio of the desired material with respect to the total material in the vessel.

        Raises
        ---------------
        None
        '''

        # get the name of the vessel object containing the target material
        label = vessel.label

        # extract the material dictionary from the vessel object
        materials = vessel._material_dict

        # obtain the names and value lists from the material dictionary
        material_names = list(materials.keys())
        values = list(materials.values())

        # get the index and then amount of the target material in the material dictionary
        target_material_index = material_names.index(desired_material)
        target_material_amount = values[target_material_index][1]

        # sum the amounts of all materials in the material dictionary
        total_material_amount = 0
        for value_list in values:
            material_amount = value_list[1]
            total_material_amount += material_amount

        # calculate the reward as the purity of the target material in the vessel
        reward = target_material_amount / total_material_amount

        # print the results to the terminal
        print("Done Reward in {}:".format(label))
        print("----- Target Material Amount = {} -----".format(target_material_amount))
        print("----- Total Material Amount = {} -----".format(total_material_amount))
        print("----- Reward = {} -----".format(reward))

        return reward

    def calc_reward(self):
        '''
        Method to calculate the total reward using all vessels from the Distillation Bench. Upon
        completing the Distillation Bench, several vessels are outputted each containing materials
        that may or may not include the desired material. No material is lost, so the amount of
        target material in the initial extraction vessel is spead about the vessels upon completing
        the Distillation Bench. The purpose of Distillation Bench is to isolate the desired material
        in its own vessel. Therefore, a greater reward should be given to a greater purity and the
        desired material being present in fewer separate vessels, similar to the reward generated
        by the Extraction Bench.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `final_reward` : `float`
            The final reward calculated from each vessel.

        Raises
        ---------------
        None 
        '''

        total_reward = 0

        # sum the rewards from each vessel containing at least some of the desired material
        for vessel in self.desired_vessels:
            total_reward += self.calc_vessel_purity(
                vessel=vessel,
                desired_material=self.desired_material
            )

        # divide the sum reward by the number of vessels containing some of the desired material
        if len(self.desired_vessels) == 0:
            final_reward = 0
        else:
            final_reward = total_reward / len(self.desired_vessels)

        return final_reward

    def validate_vessels(self, purity_threshold=0.0):
        '''
        Method to examine each vessel outputted by the Distillation Bench and determine if it has
        passed the purity threshold and should be outputted/saved.

        Parameters
        ---------------
        `purity_threshold` : `float` (default=`0.0`)
            The minimum purity a vessel must exceed to be saved and used in future processes.

        Returns
        ---------------
        `validated_vessels` : `list` (default=`[]`)
            A list of vessel objects that each contain enough of the desired material such that the
            purity of the desired material exceeds the `purity_threshold` parameter.

        Raises
        ---------------
        None
        '''

        # enusre the purity threshold is not negative and does not exceed 1.0;
        # if so, use the default purity threshold instead
        if any([
                purity_threshold < 0.0,
                purity_threshold > 1.0
        ]):
            purity_threshold = 0.0

        # create a space to store vessels with purity that exceeds the threshold
        validated_vessels = []

        # iterate through each vessel that contains at least some of the desired material
        for vessel in self.desired_vessels:

            # determine the purity reward of each vessel
            reward = self.calc_vessel_purity(
                vessel=vessel,
                desired_material=self.desired_material
            )

            # check if the purity reward exceeds the defined threshold
            if reward > purity_threshold:
                validated_vessels.append(vessel)

        return validated_vessels
