# Copyright (c) 2025 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688
from tmu.clause_bank.base_clause_bank import BaseClauseBank
from tmu.tmulib import ffi, lib
import numpy as np
from scipy.sparse import csr_matrix
import logging

LOGGER = logging.getLogger(__name__)


class ClauseBankSets(BaseClauseBank):
    def __init__(
            self,
            concept_sets,
            seed: int,
            number_of_states,
            d: float,
            batching=True,
            incremental=True,
            absorbing=-1,
            absorbing_exclude=None,
            absorbing_include=None,
            literal_sampling=1.0,
            feedback_rate_excluded_literals=1,
            literal_insertion_state=-1,
            **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.number_of_states = int(number_of_states)
        self.incremental = incremental

        self.concept_sets = concept_sets
        self.number_of_concept_sets = self.concept_sets.shape[0]
        self.number_of_elements = self.X_shape[1]

        self.number_of_features = self.number_of_concept_sets
        self.number_of_literals = self.number_of_concept_sets
        self.number_of_ta_chunks = int((self.number_of_concept_sets - 1) / 32 + 1)

        self.number_of_element_chunks = int((self.number_of_elements - 1) / 32 + 1)

        self.clause_output = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))

        self.clause_bank_included = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_concept_sets, 2),
                                                                  dtype=np.uint32))  # Contains index and state of included literals, none at start
        self.clause_bank_included_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint32))

        self.clause_bank_excluded = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_concept_sets, 2),
                                                                  dtype=np.uint32))  # Contains index and state of excluded literals
        self.clause_bank_excluded_length = np.ascontiguousarray(np.empty(self.number_of_clauses, dtype=np.uint32))  
        self.clause_bank_excluded_length[:] = int(self.number_of_concept_sets) # All literals excluded at start
        for j in range(self.number_of_clauses):
            self.clause_bank_excluded[j, :, 0] = np.arange(self.number_of_concept_sets, dtype=np.uint32)
            self.clause_bank_excluded[j, :, 1] = self.number_of_states // 2 - 1  # Initialize excluded literals in least forgotten state

        # Feature vector for calculating set intersections
        self.set_intersection = np.ascontiguousarray(np.zeros(self.number_of_element_chunks, dtype=np.uint32))

        self.encoded_concept_sets = np.ascontiguousarray(np.zeros(self.number_of_concept_sets * self.number_of_element_chunks, dtype=np.uint32))

        self.true_concept_sets = np.ascontiguousarray(np.zeros(self.number_of_concept_sets, dtype=np.uint32))

        self._cffi_init()

        lib.cbse_encode_sets(self.ptr_concept_sets_indptr, self.ptr_concept_sets_indices, self.number_of_concept_sets, self.number_of_elements, self.ptr_encoded_concept_sets)

    def _cffi_init(self):
        self.ptr_clause_bank_included = ffi.cast("unsigned int *", self.clause_bank_included.ctypes.data)
        self.ptr_clause_bank_included_length = ffi.cast("unsigned int *",
                                                        self.clause_bank_included_length.ctypes.data)
        self.ptr_clause_output = ffi.cast("unsigned int *", self.clause_output.ctypes.data)

        self.ptr_clause_bank_excluded = ffi.cast("unsigned int *", self.clause_bank_excluded.ctypes.data)
        self.ptr_clause_bank_excluded_length = ffi.cast("unsigned int *",
                                                        self.clause_bank_excluded_length.ctypes.data)
        self.ptr_set_intersection = ffi.cast("unsigned int *", self.set_intersection.ctypes.data)

        self.ptr_concept_sets_indptr = ffi.cast("unsigned int *", self.concept_sets.indptr.ctypes.data)
        self.ptr_concept_sets_indices = ffi.cast("unsigned int *", self.concept_sets.indices.ctypes.data)
        self.ptr_encoded_concept_sets = ffi.cast("unsigned int *", self.encoded_concept_sets.ctypes.data)

        self.ptr_true_concept_sets = ffi.cast("unsigned int *", self.true_concept_sets.ctypes.data)

    def calculate_clause_outputs_predict(self, encoded_X, e):       
        lib.cbse_calculate_clause_outputs(
            encoded_X[1][e],
            encoded_X[0].indptr[e + 1] - encoded_X[0].indptr[e],
            self.ptr_encoded_concept_sets,
            self.number_of_concept_sets,
            self.number_of_elements,
            self.number_of_clauses,
            self.ptr_clause_output,
            self.ptr_clause_bank_included,
            self.ptr_clause_bank_included_length,
            1
        )
 
        return self.clause_output.copy()

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        lib.cbse_calculate_clause_outputs(
            encoded_X[1][e],
            encoded_X[0].indptr[e + 1] - encoded_X[0].indptr[e],
            self.ptr_encoded_concept_sets,
            self.number_of_concept_sets,
            self.number_of_elements,
            self.number_of_clauses,
            self.ptr_clause_output,
            self.ptr_clause_bank_included,
            self.ptr_clause_bank_included_length,
            0
        )

        return self.clause_output

    def type_i_feedback(
            self,
            update_p,
            clause_active,
            literal_active,
            encoded_X,
            e
    ):
        lib.cbse_type_i_feedback(
            update_p,
            self.number_of_clauses,
            self.number_of_states,
            self.s,
            self.boost_true_positive_feedback,
            self.max_included_literals,
            ffi.cast("int *", clause_active.ctypes.data),
            encoded_X[1][e],
            encoded_X[0].indptr[e + 1] - encoded_X[0].indptr[e],
            self.ptr_encoded_concept_sets,
            self.number_of_concept_sets, 
            self.number_of_elements,
            self.ptr_set_intersection,
            self.ptr_true_concept_sets,
            self.ptr_clause_output,
            self.ptr_clause_bank_included,
            self.ptr_clause_bank_included_length,
            self.ptr_clause_bank_excluded,
            self.ptr_clause_bank_excluded_length
        )

    def type_ii_feedback(
            self,
            update_p,
            clause_active,
            literal_active,
            encoded_X,
            e
    ):
        lib.cbse_type_ii_feedback(
            update_p,
            self.number_of_clauses,
            self.number_of_states,
            ffi.cast("int *", clause_active.ctypes.data),
            encoded_X[1][e],
            encoded_X[0].indptr[e + 1] - encoded_X[0].indptr[e],
            self.ptr_encoded_concept_sets,
            self.number_of_concept_sets, 
            self.number_of_elements,
            self.ptr_set_intersection,
            self.ptr_true_concept_sets,
            self.ptr_clause_output,
            self.ptr_clause_bank_included,
            self.ptr_clause_bank_included_length,
            self.ptr_clause_bank_excluded,
            self.ptr_clause_bank_excluded_length
        )

    def number_of_include_actions(self, clause):
        return self.clause_bank_included_length[clause]

    def number_of_exclude_actions(self, clause):
        return self.clause_bank_excluded_length[clause]

    def get_ta_action(self, clause, ta):
        if ta in self.clause_bank_included[clause, :self.clause_bank_included_length[clause], 0]:
            return 1
        else:
            return 0

    def get_ta_state(self, clause, ta):
        action = self.get_ta_action(clause, ta)
        if action == 0:
            literals = self.clause_bank_excluded[clause, :self.clause_bank_excluded_length[clause], 0]
            return self.clause_bank_excluded[clause, np.nonzero(literals == ta)[0][0], 1]
        else:
            literals = self.clause_bank_included[clause, :self.clause_bank_included_length[clause], 0]
            return self.clause_bank_included[clause, np.nonzero(literals == ta)[0][0], 1]

    def prepare_X(self, X):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1), dtype=np.uint32)
        X_p = []
        for e in range(X.shape[0]):
            X_p.append(
                ffi.cast("unsigned int *", X_csr.indices[X_csr.indptr[e]:X_csr.indptr[e + 1]].ctypes.data)
            )
        return (X_csr, X_p)
