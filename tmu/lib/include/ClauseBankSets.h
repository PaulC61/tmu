/*

Copyright (c) 2025 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
https://arxiv.org/abs/1905.09688

*/

void cbse_encode_sets(
        unsigned int *sets_indptr,
        unsigned int *sets_indices,
        int number_of_sets,
        int number_of_elements,
        unsigned int *encoded_sets
);

void cbse_calculate_clause_outputs(
        unsigned int *input_set_indices,
        int input_set_number_of_indices,
        unsigned int *concept_sets,
        int number_of_concept_sets,
        int number_of_elements,
        int number_of_clauses,
        unsigned int *clause_output,
        unsigned int *set_intersection,
        unsigned int *clause_bank_included,
        unsigned int *clause_bank_included_length,
        unsigned int *empty_clause_false
);

void cbse_type_i_feedback(
    float update_p,
    float s,
    int boost_true_positive_feedback,
    int max_included_literals,
    int *clause_active,
    unsigned int *literal_active,
    unsigned int *indices,
    int number_of_indices,
    unsigned int *Xi,
    int number_of_clauses,
    int number_of_literals,
    int number_of_states,
    unsigned int *clause_bank_included,
    unsigned int *clause_bank_included_length,
    unsigned int *clause_bank_excluded,
    unsigned int *clause_bank_excluded_length
);

void cbse_type_ii_feedback(
    float update_p,
    int *clause_active,
    unsigned int *literal_active,
    unsigned int *indices,
    int number_of_indices,
    unsigned int *Xi,
    int number_of_clauses,
    int number_of_literals,
    int number_of_states,
    unsigned int *clause_bank_included,
    unsigned int *clause_bank_included_length,
    unsigned int *clause_bank_excluded,
    unsigned int *clause_bank_excluded_length
);