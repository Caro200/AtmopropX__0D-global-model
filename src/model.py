import numpy as np 
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
from scipy.interpolate import interp1d

#Local modules
from src.util import load_csv, load_cross_section
from src.auxiliary_funcs import pressure, maxwellian_flux_speed, u_B, A_eff, A_eff_1, SIGMA_I, R_ind, h_L
from src.specie import Specie, Species
from reactions.reaction import Reaction

class GlobalModel:

    def __init__(self, config_dict: dict[str, float], species: Species, reaction_set: list[Reaction]):
        """Object simulating the evolution of a plasma with 0D model. 
            Inputs :
                config_dict : dictionary containing all parameters about the experimental setup
                species : instance of Species with all species being considered
                reaction_set : list with all reactions being considered -> [Reaction]"""
        self.load_chemistry()
        self.load_config(config_dict)
        self.species = species
        self.reaction_set = reaction_set

    def load_config(self, config_dict: dict[str, float]):

        # Geometry
        self.R = config_dict['R'] # Chamber's radius 
        self.L = config_dict['L'] # Length
        
        # Neutral flow
        self.m_i = config_dict['m_i']
        self.Q_g = config_dict['Q_g']
        self.beta_g = config_dict['beta_g']
        self.kappa = config_dict['kappa']

        # Ions
        self.beta_i = config_dict['beta_i'] # Transparency to ions (!!! no exact formula => to be determined later on...)
        self.V_grid = config_dict['V_grid'] # potential difference

        # Electrical
        self.omega = config_dict['omega']  # Pulsation of electro-mag fiel ?
        self.N = config_dict['N']  # Number of spires
        self.R_coil = config_dict['R_coil'] #Radius of coil around plasma
        self.I_coil = config_dict['I_coil'] 

        # Initial values = values in steady state before ignition of plasma
        self.T_e_0 = config_dict['T_e_0']
        self.n_e_0 = config_dict['n_e_0']
        self.T_g_0 = config_dict['T_g_0']
        # gas density (in m^-3 ??) in steady state without plasma
        self.n_g_0 = pressure(self.T_g_0, self.Q_g,     
                              maxwellian_flux_speed(self.T_g_0, self.m_i),
                              self.A_g) / (k * self.T_g_0)
    
    @property
    def A_g(self): return self.beta_g * pi * self.R**2

    @property
    def A_i(self): return self.beta_i * pi * self.R**2

    @property
    def V(self): return pi * self.R**2 * self.L 

    @property
    def A(self): return 2*pi*self.R**2 + 2*pi*self.R*self.L

    @property
    def v_beam(self): 
        """Ion beam's exit speed"""
        return np.sqrt(2 * e * self.V_grid / self.m_i)

    def flux_i(self, T_e, T_g, n_e, n_g):
        """Ion flux leaving the thruster through the grid holes"""
        return h_L(n_g, self.L) * n_e * u_B(T_e, self.m_i)

    def thrust_i(self, T_e, T_g, n_e, n_g):
        """Total thrust produced by the ion beam"""
        return self.flux_i(T_e, T_g, n_e, n_g) * self.m_i * self.v_beam * self.A_i

    def j_i(self, T_e, T_g, n_e, n_g):
        """Ion current density extracted by the grids"""
        return self.flux_i(T_e, T_g, n_e, n_g) * e

    def eval_property(self, func, sol):
        """Calculates a property based on 'state' for all 't'.
            sol must be a np.array with shape (nb_of_t's, dimension_of_state) where each line represents a state"""
        prop = np.zeros(sol.shape[0])   #number of instants
        for i in np.arange(sol.shape[0]):
            prop[i] = func(sol[i])
        return prop

    # def P_loss(self, state):
    #     """
    #     Calcule les pertes d'énergie des électrons :
    #     - Ionisation
    #     - Excitation
    #     - Collisions élastiques
    #     - Pertes aux parois
    #     - Dissociation et excitation vibrationnelle (plasma d'air)
    #     """
    #     # Ionisation et excitation
    #     a = self.E_iz * n_e * n_g * self.K_iz(T_e)
    #     b = self.E_ex * n_e * n_g * self.K_ex(T_e)
        
    #     # Collisions élastiques (énergie transférée au gaz)
    #     c = 3 * (m_e / self.m_i) * k * (T_e - T_g) * n_e * n_g * self.K_el(T_e)

    #     # Pertes aux parois
    #     d = 7 * k * T_e * n_e * u_B(T_e, self.m_i) * A_eff(n_g, self.R, self.L) / self.V

    #     # Dissociation et excitation vibrationnelle 
    #     K_diss = self.K_diss(T_e)  # Taux de dissociation

    #     e = self.E_diss * n_e * n_g * K_diss
    #     f = E_vibr * n_e * n_g * K_vibr

    #     return a + b + c + d + e + f


    

    def P_abs(self, state):
        # ! n_g à changer
        return R_ind(self.R, self.L, self.N, self.omega, n_e, n_g, self.K_el(T_e)) * self.I_coil**2 / 2 # the original code divided by V : density of power ?
    
    def electron_energy_derivative(self, state):
        """Returns the derivative of electron energy : d/dt (3/2 n_e e T_e)

        Input :
        'state' has format [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato] """
        # ! losses with walls not considered
        p_loss = 0
        for reac in self.reaction_set:
                p_loss += reac.electron_loss_power(state)      


        """ pertes au niveaux des parois """
        def S_eff(self): # surface efficace de pertes
            return 2 * pi * self.R^2 * h_R(self,R) + 2 * pi * self.R * self.L * h_L(self,L)
        T_e = state[Species.nb()] 
        E_cin = 7 * e * T_e # Dans le modèle de Chabert ; cas général plus compliqué
        v_e = maxwellian_flux_speed(T_e,m_e) 
        Gamma_e = n_e v_e / 4# Flux electronique vers les parois
        P_wall = E_cin * Gamma_e * S_eff(self) / V(self) # Puissance volumique perdue aux parois
        """ balance énergétique """
        power_balance = self.P_abs(state) - p_loss - P_wall

        
        power_balance = self.P_abs(state) - p_loss

        return power_balance


    def monoatomic_gas_energy_derivative(self, state):
        """
        Calcule la dérivée de l'énergie du gaz : (3/2) * n_g * k_B * T_g
        
        Input :
        'state' has format [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
        """

        # Taux de réaction
        K_diss = self.K_diss(T_e)  # Taux de dissociation
        K_vibr = self.K_vibr(T_e)  # Taux d'excitation vibrationnelle
        K_rot = self.K_rot(T_e)  # Taux d'excitation rotationnelle

        # Collisions élastiques : transfert d'énergie des électrons vers le gaz
        a = 3 * (m_e / self.m_i) * k * (T_e - T_g) * n_e * n_g * self.K_el(T_e)

        # Transfert d'énergie des ions au gaz neutre via collisions
        b = (1/4) * self.m_i * (u_B(T_e, self.m_i)**2) * n_e * n_g * SIGMA_I * maxwellian_flux_speed(T_g, self.m_i)

        # Dissociation des molécules 
        c = E_diss * n_e * n_g * K_diss

        # Excitation vibrationnelle 
        d = E_vibr * n_e * n_g * K_vibr

        # Excitation rotationnelle 
        e = E_rot * n_e * n_g * K_rot

        # Transfert de chaleur aux parois : calcul de lamda0 ? 
        lambda_0 = self.R / 2.405 + self.L / pi  # Longueur de diffusion thermique
        f = self.kappa * (T_g - self.T_g_0) * self.A / (self.V * lambda_0)

        # Somme des contributions
        return a + b + c + d + e - f


    def particles_densities_derivative(self, state: NDArray[float]): # type: ignore
        """Takes the state as input and returns derivative of all particle densities
            Input :
            state : describes state. Has format [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato] """
        dn = np.zeros(self.species.nb)
        for reac in self.reaction_set:
            dn += reac.density_change_rate(state)
        # dn[2] = inflow_of_N2 ...
        return dn
    
    # def energies_derivatives(self, state: NDArray[float]): # type: ignore
    #     """Takes the state as input and returns derivative of all energies
    #         Input :
    #         state : describes state. Has format [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato] """
    #     nb_e = state.shape[0] - self.species.nb
    #     dE = np.zeros(nb_e) 
    #     #pass
    #     return dE    


    def P_rf(self, state: NDArray[float]): # type: ignore
        """Calculates the power delivered by the coil using RF"""
        # ! To adapt to new var state
        T_e, T_g, n_e, n_g = state
        R_ind_val = R_ind(self.R, self.L, self.N, self.omega, n_e, n_g, self.K_el(T_e))
        return (1/2) * (R_ind_val + self.R_coil) * self.I_coil**2

    def f_dy(self, t, state):
        # ! Dérivée en température !!!
        """Returns the derivative of the vector 'state' describing the state of plasma.
            'state' has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        dy = np.zeros(state.shape)

        dy[:self.species.nb] = self.particles_densities_derivative(state)
        dy[self.species.nb] = self.electron_energy_derivative(state)
        dy[self.species.nb] = self.monoatomic_gas_energy_derivative(state)

        return dy
    

    def solve(self, t0, tf):
        y0 = np.array([self.T_e_0, self.T_g_0, self.n_e_0, self.n_g_0])
        return solve_ivp(self.f_dy, (t0, tf), y0, method='LSODA')


    def solve_for_I_coil(self, I_coil):
        """Calculates for a list of intensity in the coil the resulting power consumption and the resulting thrust.
            ## Returns
            power_array , list_of(`state` after long time)"""
        p = np.zeros(I_coil.shape[0])
        solution = np.zeros((I_coil.shape[0], 4))  #shape = (y,x)

        for i, I in enumerate(I_coil):
            self.I_coil = I

            sol = self.solve(0, 5e-2)    # TODO Needs some testing

            final_state = sol.y[:, -1]

            p[i] = self.P_rf(final_state)

            solution[i] = final_state
            
        return p, solution
