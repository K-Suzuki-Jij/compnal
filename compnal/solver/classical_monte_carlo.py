from typing import Optional, Union

from compnal.base_compnal import base_solver
from compnal.model.polynomial_ising import PolynomialIsing
from compnal.model.ising import Ising
from compnal.solver.algorithm import Algorithm, cast_base_algorithm, cast_algorithm

ModelType = Union[PolynomialIsing, Ising]


class ClassicalMonteCarlo:
    """ClassicalMonteCarlo class. This class can treat classical models.

    Attributes:
        algorithm (Algorithm): The algorithm of update method.
        num_sweeps (int): The number of sweeps.
        num_samples (int): The number of samples.
        num_threads (int): The number of threads.
        temperature (float): The temperature.
        model (ModelType): The model.
    """

    def __init__(self, 
        model: ModelType
        ) -> None:
        """The constructor.

        Args:
            model (ModelType): The classical model.
            algorithm (Algorithm, optional): The algorithm of update method. Defaults to Algorithm.METROPOLIS.
        """

        self.__base_solver = base_solver.make_classical_monte_carlo(
            model=model._base_model
        )

        self.__model = model

    def get_seed(self) -> int:
        """Get the seed used by the monte carlo algorithm.

        Returns:
            int: The seed.
        """
        return self.__base_solver.get_seed()

    def get_samples(self) -> list[list[int]]:
        """Get the samples.

        Returns:
            list[list[int]]: The samples.
        """
        return self.__base_solver.get_samples()

    def run(self, seed: Optional[int] = None) -> None:
        """Run the monte carlo algorithm.

        Args:
            seed (Optional[int], optional): The seed used by the monte carlo algorithm. Defaults to None.
        """
        if seed is None:
            self.__base_solver.run()
        else:
            self.__base_solver.run(seed=seed)

    def calculate_average(self) -> float:
        """Calculate average of all the samples.

        Returns:
            float: The average value.
        """
        return self.__base_solver.calculate_average()

    def calculate_onsite_average(self) -> list[float]:
        """Calculate average of samples at each sites.

        Returns:
            list[float]: Average of samples at each sites.
        """
        return self.__base_solver.calculate_onsite_average()

    def calculate_moment(self, degree: int) -> float:
        """Calculate the moment with the specified degree of each sample and take average for all the moments.

        Args:
            degree (int): The degree of the moment.

        Returns:
            float: The moment.
        """
        return self.__base_solver.calculate_moment(degree=degree)

    def calculate_correlation(
        self,
        origin: Union[int, list[Union[int, str, list[Union[int, str]]]]],
        index_list: Union[
            list[int], list[list[Union[int, str, list[Union[int, str]]]]]
        ],
    ) -> list[float]:
        """Calculate correlation function list.

        Args:
            origin (Union[int, list[Union[int, str, list[Union[int, str]]]]]): The index.
            index_list (Union[list[int], list[list[Union[int, str, list[Union[int, str]]]]]]): The index list.

        Returns:
            list[float]: Correlation function list from the samples at the sites of origin and index in index_list.
        """
        return self.__base_solver.calculate_correlation(origin, index_list)

    @property
    def algorithm(self) -> Algorithm:
        return cast_base_algorithm(self.__base_solver.get_algorithm())

    @algorithm.setter
    def algorithm(self, algorithm: Algorithm) -> None:
        self.__base_solver.set_algorithm(cast_algorithm(algorithm))

    @property
    def num_sweeps(self) -> int:
        return self.__base_solver.get_num_sweeps()

    @num_sweeps.setter
    def num_sweeps(self, num_sweeps: int) -> None:
        self.__base_solver.set_num_sweeps(num_sweeps=num_sweeps)

    @property
    def num_samples(self) -> int:
        return self.__base_solver.get_num_samples()

    @num_samples.setter
    def num_samples(self, num_samples: int) -> None:
        self.__base_solver.set_num_samples(num_samples=num_samples)

    @property
    def num_threads(self) -> int:
        return self.__base_solver.get_num_threads()

    @num_threads.setter
    def num_threads(self, num_threads: int) -> None:
        self.__base_solver.set_num_threads(num_threads=num_threads)

    @property
    def temperature(self) -> float:
        return self.__base_solver.get_temperature()

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        self.__base_solver.set_temperature(temperature=temperature)

    @property
    def model(self) -> ModelType:
        return self.__model
