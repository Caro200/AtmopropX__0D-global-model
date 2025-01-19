from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from specie import Specie, Species


class Reaction:
    """
        Reaction class
            /!\ Electrons should be added to reactives and products only if they are not spectators (otherwise pbs with density_rate_change)
            Inputs : 
                species : instance of class Species, lists all species present 
                reactives : list with all reactives names
                products : list with all products names
                rate_constant : function taking as arguments (T_g, T_e, )
                stoechio_coeffs : stoechiometric coefficients always positive
    """

    def __init__(self, species: Species, reactives: list[str], products: list[str], rate_constant, energy_treshold: float, stoechio_coeffs: list[float]=None, spectators: list[str]=None):
        """
        Reaction class
        /!\ Electrons should be added to reactives and products only if they are not spectators (otherwise pbs with density_rate_change)
            Inputs : 
                species : instance of class Species, lists all species present 
                reactives : list with all reactives names
                products : list with all products names
                rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
                stoechio_coeffs : stoechiometric coefficients always positive"""
        self.species = species

        self.reactives: list[Specie] = [self.species.get_specie_by_name(name) for name in reactives]
        self.reactives_indices = [self.species.get_index_by_instance(sp) for sp in self.reactives]
        assert max(self.reactives_indices) < self.species.nb , "Reactive index is greater than number of species"

        self.products: list[Specie] = [self.species.get_specie_by_name(name) for name in products]
        self.products_indices = [self.species.get_index_by_instance(sp) for sp in self.products]
        assert max(self.products_indices) < self.species.nb , "Product index is greater than number of species"


        if stoechio_coeffs :
            self.stoechio_coeffs = np.array(stoechio_coeffs)
        else:
            # sets stoechio_coeffs at 1 for reactives and products if not defined
            self.stoechio_coeffs = np.zeros(self.species.nb)
            for i in self.reactives_indices:
                self.stoechio_coeffs[i] = 1
            for sp in self.products_indices:
                self.stoechio_coeffs[i] = 1

        self.energy_threshold = energy_treshold
        self.rate_constant = rate_constant     # func
        self.spectators = spectators
        

    def density_change_rate(self, state: NDArray[float]): # type: ignore
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        K = self.rate_constant(state[self.species.nb:])
        product = K * np.prod(state[self.reactives_indices]) # product of rate constant and densities of all the stuff
        rate = np.zeros(self.species.nb)
        for sp in self.reactives:
            i = self.species.get_index_by_instance(sp)
            rate[i] = - product * self.stoechio_coeffs[i]
        for sp in self.products:
            i = self.species.get_index_by_instance(sp)
            rate[i] = + product * self.stoechio_coeffs[i]
        
        return rate


    def electron_loss_power(self, state: NDArray[float]): # type: ignore
        """Function meant to return the change in energy due to this specific equation.
            HOWEVER : seems like it is necessary to account for difference in temperature of atoms molecules and electrons...
            Thus 1 function per "Temperatur type" will be needed"""
        K = self.rate_constant(state)
        power_loss = self.energy_threshold * K * np.prod(state[self.reactives_indices]) # product of energy, rate constant and densities of all the stuff
        
        return power_loss

    def diatomic_gas_energy_change_rate(self, state: NDArray[float]):
        """ When the reactions will be sorted, we will add here the term E_diss * K_diss * n1 * n2 """
        return 0
    
    
    def __str__(self):
        """Returns string describing the reaction"""
        def format_species(species, species_indices):
            terms = []
            for idx, sp in zip(species_indices, species):
                coeff = self.stoechio_coeffs[idx]
                # Format coefficient: display as integer if it is a whole number, else as float with 2 decimals
                if coeff.is_integer():
                    coeff_str = f"{int(coeff)}" if coeff != 1 else ""
                else:
                    coeff_str = f"{coeff:.2f}" 
                term = f"{coeff_str} {sp.name}".strip()
                terms.append(term)
            return " + ".join(terms)

        reactives_str = format_species(self.reactives, self.reactives_indices)
        products_str = format_species(self.products, self.products_indices)
        return f"{reactives_str} -> {products_str}          K_r = {self.rate_constant.__name__}"



if __name__ == "__main__":
    def K_diss_I2(Ts):
        print("Temperatures : ",Ts)
        return 2
    
    species_list = Species([Specie("I0", 10.57e-27, 0), Specie("I1", 10.57e-27, 0), Specie("I2", 10.57e-27, 0), Specie("I3", 10.57e-27, 0), Specie("I4", 10.57e-27, 0), Specie("I5", 10.57e-27, 0)])

    reac = Reaction(species_list, ["I2", "I4"], ["I5"], K_diss_I2, 10)
    state = np.array([1,2,3,4,5,6, -181,-182]) # jusqu'à 6 = densité, apres T°
    print(reac.density_change_rate(state))
    print(reac)




class ElasticCollisionWithElectron(Reaction):

    def __init__(self, species: Species, colliding_specie: str, rate_constant, energy_treshold: float):
        super().__init__(species, [colliding_specie], [colliding_specie], rate_constant, energy_treshold)

    @override
    def density_change_rate(self, state):
        return np.zeros(self.species.nb)


    @override
    def electron_loss_power(self, state):
        K = self.rate_constant(state)
        mass_ratio = m_e / self.reactives[0].mass
        delta_temp = state[self.species.nb] - state[self.species.nb + self.reactives[0].nb_atoms ]

        energy_change = 3 * mass_ratio * k_B * delta_temp * state[0] * state[self.reactives_indices[0]] * K 
   
        return energy_change

    @override
    def diatomic_gas_energy_change_rate(self, state):
        if self.reactives[0].nb_atoms == 2:
            K = self.rate_constant(state)
            mass_ratio = m_e / self.reactives[0].mass
            delta_temp = state[self.species.nb] - state[self.species.nb + self.reactives[0].nb_atoms ]

            energy_change = 3 * mass_ratio * k_B * delta_temp * state[0] * state[self.reactives_indices[0]] * K 
        else:
            energy_change = 0
        return energy_change
