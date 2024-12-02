import heapq

#TODO: - Modify constrains for BCI use case
#      - Use class for structured inference, implement "structure forward pass" in model class

class BeamSearch:
    def __init__(self, sequence_model, beam_width, max_steps):
        """
        Initializes the BeamSearch class.

        Args:
            sequence_model (function): Function to generate next states and probabilities.
            beam_width (int): Number of sequences to retain at each step.
            max_steps (int): Maximum number of steps to run the search.
        """
        self.sequence_model = sequence_model
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.constraints = []  # List of constraint functions

    def add_constraint(self, constraint_fn):
        """
        Adds a constraint function to the list of constraints.

        Args:
            constraint_fn (function): Function that takes a sequence and returns True if valid.
        """
        self.constraints.append(constraint_fn)

    def apply_constraints(self, sequence):
        """
        Applies all constraints to a sequence.

        Args:
            sequence (list): The sequence to validate.

        Returns:
            bool: True if the sequence satisfies all constraints, False otherwise.
        """
        return all(constraint(sequence) for constraint in self.constraints)

    def search(self, start_state):
        """
        Performs beam search.

        Args:
            start_state (list): The initial state or token to start the search.

        Returns:
            List[tuple]: The best sequences and their scores [(sequence, score), ...].
        """
        beam = [(start_state, 1.0)]  # Initialize beam with the starting state

        for _ in range(self.max_steps):
            all_candidates = []

            # Expand each sequence in the beam
            for sequence, score in beam:
                next_states = self.sequence_model(sequence)

                for next_state, prob in next_states:
                    new_sequence = sequence + [next_state]
                    new_score = score * prob

                    # Apply constraints (if any)
                    if not self.constraints or self.apply_constraints(new_sequence):
                        all_candidates.append((new_sequence, new_score))

            # Keep the top `beam_width` sequences
            beam = heapq.nlargest(self.beam_width, all_candidates, key=lambda x: x[1])

        return beam

    # Example constraint methods (optional, users can define their own constraints)
    def add_max_length_constraint(self, max_length):
        """
        Adds a constraint to limit the maximum length of sequences.

        Args:
            max_length (int): The maximum allowed sequence length.
        """
        self.add_constraint(lambda sequence: len(sequence) <= max_length)

    def add_no_repeats_constraint(self):
        """
        Adds a constraint to prevent repeated states in the sequence.
        """
        self.add_constraint(lambda sequence: len(sequence) == len(set(sequence)))

    def add_end_token_constraint(self, end_token):
        """
        Adds a constraint to ensure the sequence ends with a specific token.

        Args:
            end_token (str): The required ending token.
        """
        self.add_constraint(lambda sequence: sequence[-1] == end_token if len(sequence) > 0 else False)
