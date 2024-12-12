import heapq

#TODO: - Modify constrains for BCI use case
#      - Use class for structured inference, implement "structure forward pass" in model class


class BeamSearch:
    def __init__(self, beam_width, max_steps):
        """
        Initializes the BeamSearch class.

        Args:
            beam_width (int): Number of sequences to retain at each step.
            max_steps (int): Maximum number of steps to run the search.
        """
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.constraints = []  # List of constraint functions
        self.penalty_fn = None  # Function to apply penalties to scores (aka soft constraints)

    def add_constraint(self, constraint_fn):
        """
        Adds a constraint function to the list of constraints.

        Args:
            constraint_fn (function): Function that takes a sequence and returns True if valid.
        """
        self.constraints.append(constraint_fn)

    def add_penalty(self, penalty_fn):
        """
        Adds a penalty function to apply to scores.

        Args:
            penalty_fn (function): Function that takes a sequence and returns a penalty score.
        """
        self.penalty_fn = penalty_fn

    def apply_constraints(self, sequence):
        """
        Applies all constraints to a sequence.

        Args:
            sequence (list): The sequence to validate.

        Returns:
            bool: True if the sequence satisfies all constraints, False otherwise.
        """
        return all(constraint(sequence) for constraint in self.constraints)

    def search(self, prob_matrix):
        """
        Performs beam search.

        Args:
            prob_matrix (np.ndarray): The 17x16 matrix representing the probabilities at each timestep.

        Returns:
            List[tuple]: The best sequences and their scores [(sequence, score), ...].
        """
        beam = [([], 1.0)]  # Initialize beam with the starting state (empty sequence, score = 1)

        # Iterate through each timestep
        for step in range(self.max_steps):
            all_candidates = []

            # Expand each sequence in the beam
            for sequence, score in beam:
                if step < prob_matrix.shape[1]:  # Ensure we don't exceed the matrix dimensions
                    next_state_probs = prob_matrix[:, step]  # Probabilities for the current timestep

                    # For each possible next state (label) at this step
                    for next_state, prob in enumerate(next_state_probs):
                        new_sequence = sequence + [next_state]
                        new_score = score * prob

                        # Apply constraints (if any)
                        if not self.constraints or self.apply_constraints(new_sequence):
                            if self.penalty_fn:
                                penalty = self.penalty_fn(self,new_sequence)
                                new_score *= penalty
                            all_candidates.append((new_sequence, new_score))

            # Keep the top `beam_width` sequences
            beam = heapq.nlargest(self.beam_width, all_candidates, key=lambda x: x[1])

        # Return the top sequences and their scores
        return [(seq, score) for seq, score in beam]

    # Non-repeating/consistency penalty
    def non_repeating_penalty(self, sequence):
        unique_values = len(set(sequence))
        total_values = len(sequence)
        penalty = 1.0 / (1.0 + unique_values / total_values)  # Scaled penalty
        return penalty