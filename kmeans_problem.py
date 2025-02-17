from qubots.base_problem import BaseProblem
import random
import os

class KMeansProblem(BaseProblem):
    """
    K-Means Clustering Problem for Qubots.
    
    Given a set of observations along several dimensions, the goal is to partition these observations
    into k clusters so as to minimize the total variance. The variance of a cluster is defined as the
    sum of squared Euclidean distances between each observation in the cluster and the cluster's centroid.
    
    **Candidate Solution Representation:**
      A list of length k, where each element is a list of observation indices (0-indexed) assigned to that cluster.
    """
    
    def __init__(self, instance_file: str, k: int = 2, **kwargs):
        self.k = k
        self.nb_observations, self.nb_dimensions, self.coordinates_data = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):
        """
        Reads the instance file.
        
        The instance file format:
          - First line: two integers: number of observations and number of dimensions.
          - For each observation: a line with the coordinate along each dimension followed by a label (which is ignored).
        """

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, "r") as f:
            lines = f.readlines()
        # Remove empty lines.
        lines = [line.strip() for line in lines if line.strip()]
        first_line = lines[0].split()
        nb_observations = int(first_line[0])
        nb_dimensions = int(first_line[1])
        coordinates_data = []
        # Observations are assumed to be on the next nb_observations lines.
        for i in range(1, nb_observations + 1):
            parts = lines[i].split()
            # Use the first nb_dimensions values as coordinates.
            coords = [float(parts[d]) for d in range(nb_dimensions)]
            coordinates_data.append(coords)
        return nb_observations, nb_dimensions, coordinates_data
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution: a list of length k, where each element is a list of observation indices (0-indexed)
                    that belong to that cluster.
        
        Returns:
          The total variance (sum of squared Euclidean distances from each observation to its cluster centroid).
          If a cluster is empty, its variance is defined as 0.
        """
        total_variance = 0.0
        for cluster in solution:
            if len(cluster) == 0:
                continue  # An empty cluster contributes zero variance.
            # Compute the centroid for this cluster.
            centroid = [0.0] * self.nb_dimensions
            for idx in cluster:
                for d in range(self.nb_dimensions):
                    centroid[d] += self.coordinates_data[idx][d]
            size = len(cluster)
            centroid = [x / size for x in centroid]
            # Sum squared distances.
            variance = 0.0
            for idx in cluster:
                dist2 = 0.0
                for d in range(self.nb_dimensions):
                    diff = self.coordinates_data[idx][d] - centroid[d]
                    dist2 += diff * diff
                variance += dist2
            total_variance += variance
        return total_variance
    
    def random_solution(self):
        """
        Generates a random candidate solution.
        
        Randomly assigns each observation to one of the k clusters.
        """
        clusters = [[] for _ in range(self.k)]
        for idx in range(self.nb_observations):
            clusters[random.randrange(self.k)].append(idx)
        return clusters
