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

#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcount __popcnt
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"
#include <stdint.h>

void cbse_encode_sets(
        unsigned int *sets_indptr,
        unsigned int *sets_indices,
        int number_of_sets,
        int number_of_elements,
        unsigned int *encoded_sets
)
{
    unsigned int number_of_element_chunks = (number_of_elements-1)/32 + 1;

    for (int i = 0; i < number_of_sets; ++i) {
        for (int j = sets_indptr[i]; j < sets_indptr[i+1]; ++j) {
            unsigned int element_chunk = sets_indices[j] / 32;
            unsigned int element_pos = sets_indices[j] % 32;
            encoded_sets[i*number_of_element_chunks + element_chunk] |= (1U << element_pos);
        }
    }
}

void cbse_calculate_clause_outputs(
        unsigned int *input_set_indices,
        int input_set_number_of_indices,
        unsigned int *concept_sets,
        int number_of_concept_sets,
        int number_of_elements,
        int number_of_clauses,
        unsigned int *clause_output,
        unsigned int *clause_bank_included,
        unsigned int *clause_bank_included_length,
        unsigned int empty_clause_false
)
{
    unsigned int number_of_element_chunks = (number_of_elements-1)/32 + 1;

    // Evaluate each clause
    for (int j = 0; j < number_of_clauses; ++j) {
        ////printf("Clause %d\n", j);
        if (empty_clause_false && clause_bank_included_length[j] == 0) {
            clause_output[j] = 0;
            continue;
        }

        // Calculate the size of the intersection between the input set and the sets included by the clause
        int matches = 0;

        // Go through the elements in the input set, one element at a time
        for (int k = 0; k < input_set_number_of_indices; ++k) {
            unsigned int element_chunk = input_set_indices[k] / 32;
            unsigned int element_pos = input_set_indices[k] % 32;
            
            ////printf("\tElement %d\n", input_set_indices[k]);

            // Check whether the input element is an element in all of the sets included by the clause (a match)
            unsigned int match = 1;
            for (int l = 0; l < clause_bank_included_length[j]; ++l) {
                unsigned int concept_set = clause_bank_included[j*number_of_concept_sets*2 + l*2];

                //printf("\t\tConcept %d (%d)\n", concept_set, clause_bank_included[j*number_of_concept_sets*2 + l*2 + 1]);
                if ((concept_sets[concept_set*number_of_element_chunks + element_chunk] & (1U << element_pos)) == 0) {
                    match = 0;
                    //printf("\t\t\tMismatch\n");
                    break;
                }
                //printf("\t\t\tMatch\n");
            }

            matches += match;
        }

        //printf("Matches %d\n", matches);

        //fflush(stdout);
        
        //clause_output[j] = (matches > 0);
        clause_output[j] = matches;
    }
}

void cbse_type_i_feedback(
        float update_p,
        int number_of_clauses,
        int number_of_states,
        float s,
        int boost_true_positive_feedback,
        int max_included_literals,
        int *clause_active,
        unsigned int *input_set_indices,
        int input_set_number_of_indices,
        unsigned int *concept_sets,
        int number_of_concept_sets,
        int number_of_elements,
        unsigned int *set_intersection,
        unsigned int *true_concept_sets,
        unsigned int *clause_output,
        unsigned int *clause_bank_included,
        unsigned int *clause_bank_included_length,
        unsigned int *clause_bank_excluded,
        unsigned int *clause_bank_excluded_length
)
{
	unsigned int number_of_element_chunks = (number_of_elements-1)/32 + 1;

    //update_p = 0.25;

    for (int j = 0; j < number_of_clauses; ++j) {
        if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
            //printf("Type I - skip clause %d %d %d\n", j, clause_output[j], !clause_active[j]);
            //fflush(stdout);
			continue;
		}

        int clause_pos_base = j*number_of_concept_sets*2;

        if (clause_output[j] && (clause_bank_included_length[j] <= max_included_literals)) {
            // Clause True

            //printf("Type I - Clause %d True\n", j);

            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
                 int clause_included_pos = clause_pos_base + k*2;

                 //printf("\tI: %d (%d)\n", clause_bank_included[clause_included_pos], clause_bank_included[clause_included_pos + 1]);
            }

            for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
                 int clause_excluded_pos = clause_pos_base + k*2;

                 //printf("\tE: %d (%d)\n", clause_bank_excluded[clause_excluded_pos], clause_bank_excluded[clause_excluded_pos + 1]);
            }

            // Calculate intersection of input and included sets...
            // Go through the elements in the input set, one element at a time,
            // and calculate intersection of input set and included concept sets
            for (int k = 0; k < input_set_number_of_indices; ++k) {
                unsigned int element_chunk = input_set_indices[k] / 32;
                unsigned int element_pos = input_set_indices[k] % 32;
                
                // Check whether the input element is an element in all of the sets included by the clause (a match)
                unsigned int match = 1;
                for (int l = 0; l < clause_bank_included_length[j]; ++l) {
                    unsigned int concept_set = clause_bank_included[clause_pos_base + l*2];

                    if ((concept_sets[concept_set*number_of_element_chunks + element_chunk] & (1U << element_pos)) == 0) {
                        match = 0;
                        break;
                    }
                }

                if (match) {
                    set_intersection[element_chunk] |= (1U << element_pos);
                }
            }

            // Go through excluded concept sets and determine overlap with intersection of input and included concept sets
            // If there is an overlap, set the excluded concept set to True. Otherwise, set it to False
            for (int l = 0; l < clause_bank_excluded_length[j]; ++l) {
                unsigned int concept_set = clause_bank_excluded[clause_pos_base + l*2];

                true_concept_sets[concept_set] = 0;
                for (int k = 0; k < input_set_number_of_indices; ++k) {
                    unsigned int element = input_set_indices[k];
                    unsigned int element_chunk = element / 32;
                    unsigned int element_pos = element % 32;

                    if (((concept_sets[concept_set*number_of_element_chunks + element_chunk] & (1U << element_pos)) > 0) &&
                        ((set_intersection[element_chunk] & (1U << element_pos)) > 0)) {
                        true_concept_sets[concept_set] = 1;
                        break;
                    }
                }
            }

            // Remove elements from set intersection
            for (int k = 0; k < input_set_number_of_indices; ++k) {
                unsigned int element_chunk = input_set_indices[k] / 32;
                unsigned int element_pos = input_set_indices[k] % 32;

                set_intersection[element_chunk] &= ~(1U << element_pos);
            }

            // Give Type I Feedback based on true_concept_sets

            // Update state of included literals
            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
                int clause_included_pos = clause_pos_base + k*2;
 
                if (clause_bank_included[clause_included_pos + 1] < number_of_states-1 && (boost_true_positive_feedback || (((float)fast_rand())/((float)FAST_RAND_MAX) > 1.0/s))) {
                    clause_bank_included[clause_included_pos + 1] += 1;
                }
            }

            // Update state of excluded literals
            for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
                int clause_excluded_pos = clause_pos_base + k*2;
               
                if (true_concept_sets[clause_bank_excluded[clause_excluded_pos]]) {
                    if (boost_true_positive_feedback || (((float)fast_rand())/((float)FAST_RAND_MAX) > 1.0/s)) {
                        clause_bank_excluded[clause_excluded_pos + 1] += 1;
                    }
                } else if ((clause_bank_excluded[clause_excluded_pos + 1] > 1) && (((float)fast_rand())/((float)FAST_RAND_MAX) <= 1.0/s)) {
                    clause_bank_excluded[clause_excluded_pos + 1] -= 1;
                }
            }

            // Update lists
            int k = clause_bank_included_length[j];
            while (k--) {
                int clause_included_pos = clause_pos_base + k*2;

                if (clause_bank_included[clause_included_pos + 1] < number_of_states / 2) {
                    int clause_excluded_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                    clause_bank_excluded[clause_excluded_pos] = clause_bank_included[clause_included_pos];
                    clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_included[clause_included_pos + 1];
                    clause_bank_excluded_length[j] += 1;

                    clause_bank_included_length[j] -= 1;
                    int clause_included_end_pos = clause_pos_base + clause_bank_included_length[j]*2;
                    clause_bank_included[clause_included_pos] = clause_bank_included[clause_included_end_pos];       
                    clause_bank_included[clause_included_pos + 1] = clause_bank_included[clause_included_end_pos + 1];
                }
            }

            k = clause_bank_excluded_length[j];
            while (k--) {
                int clause_excluded_pos = clause_pos_base + k*2;

                if (clause_bank_excluded[clause_excluded_pos + 1] >= number_of_states / 2) {
                    int clause_included_pos = clause_pos_base + clause_bank_included_length[j]*2;
                    clause_bank_included[clause_included_pos] = clause_bank_excluded[clause_excluded_pos];
                    clause_bank_included[clause_included_pos + 1] = clause_bank_excluded[clause_excluded_pos + 1];
                    clause_bank_included_length[j] += 1;

                    clause_bank_excluded_length[j] -= 1;
                    int clause_excluded_end_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                    clause_bank_excluded[clause_excluded_pos] = clause_bank_excluded[clause_excluded_end_pos];
                    clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_excluded[clause_excluded_end_pos + 1];
                } 
            }

            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
                 int clause_included_pos = clause_pos_base + k*2;

                 //printf("\t>>I: %d (%d)\n", clause_bank_included[clause_included_pos], clause_bank_included[clause_included_pos + 1]);
            }

            for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
                 int clause_excluded_pos = clause_pos_base + k*2;

                 //printf("\t>>E: %d (%d)\n", clause_bank_excluded[clause_excluded_pos], clause_bank_excluded[clause_excluded_pos + 1]);
            }
        
            //fflush(stdout);
        } else {
            // Clause False

            //printf("Type I - Clause %d False\n", j);
            

            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
                 int clause_included_pos = clause_pos_base + k*2;

                 //printf("\tI: %d (%d)\n", clause_bank_included[clause_included_pos], clause_bank_included[clause_included_pos + 1]);
            }

            for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
                 int clause_excluded_pos = clause_pos_base + k*2;

                 //printf("\tE: %d (%d)\n", clause_bank_excluded[clause_excluded_pos], clause_bank_excluded[clause_excluded_pos + 1]);
            }

            // Type Ib Feedback

            // Update state of included literals
            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
                int clause_included_pos = clause_pos_base + k*2;

                if (((float)fast_rand())/((float)FAST_RAND_MAX) <= 1.0/s) {
                    clause_bank_included[clause_included_pos + 1] -= 1;
                }
            }

            // Update state of excluded literals
            for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
                int clause_excluded_pos = clause_pos_base + k*2;
               
                if ((clause_bank_excluded[clause_excluded_pos + 1] > 1) && (((float)fast_rand())/((float)FAST_RAND_MAX) <= 1.0/s)) {
                    clause_bank_excluded[clause_excluded_pos + 1] -= 1;
                }
            }
        
            // Update lists
            int k = clause_bank_included_length[j];
            while (k--) {
                int clause_included_pos = clause_pos_base + k*2;
                
                if (clause_bank_included[clause_included_pos + 1] < number_of_states / 2) {
                    int clause_excluded_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                    clause_bank_excluded[clause_excluded_pos] = clause_bank_included[clause_included_pos];
                    clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_included[clause_included_pos + 1];
                    clause_bank_excluded_length[j] += 1;

                    clause_bank_included_length[j] -= 1;
                    int clause_included_end_pos = clause_pos_base + clause_bank_included_length[j]*2;
                    clause_bank_included[clause_included_pos] = clause_bank_included[clause_included_end_pos];       
                    clause_bank_included[clause_included_pos + 1] = clause_bank_included[clause_included_end_pos + 1];
                }  
            }

            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
                 int clause_included_pos = clause_pos_base + k*2;

                 //printf("\t>>I: %d (%d)\n", clause_bank_included[clause_included_pos], clause_bank_included[clause_included_pos + 1]);
            }

            for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
                 int clause_excluded_pos = clause_pos_base + k*2;

                 //printf("\t>>E: %d (%d)\n", clause_bank_excluded[clause_excluded_pos], clause_bank_excluded[clause_excluded_pos + 1]);
            }
        
            //fflush(stdout);
        }
    }
}

void cbse_type_ii_feedback(
        float update_p,
        int number_of_clauses,
        int number_of_states,
        int *clause_active,
        unsigned int *input_set_indices,
        int input_set_number_of_indices,
        unsigned int *concept_sets,
        int number_of_concept_sets,
        int number_of_elements,
        unsigned int *set_intersection,
        unsigned int *true_concept_sets,
        unsigned int *clause_output,
        unsigned int *clause_bank_included,
        unsigned int *clause_bank_included_length,
        unsigned int *clause_bank_excluded,
        unsigned int *clause_bank_excluded_length
)
{
    unsigned int number_of_element_chunks = (number_of_elements-1)/32 + 1;

    //update_p = 1.0;

    for (int j = 0; j < number_of_clauses; ++j) {
        if ((!clause_output[j]) || (((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
            //printf("Type II - skip clause %d %d %d\n", j, clause_output[j], !clause_active[j]);
            //fflush(stdout);
            continue;
        }

        int clause_pos_base = j*number_of_concept_sets*2;

        //printf("Type II - Clause %d True\n", j);

        for (int k = 0; k < clause_bank_included_length[j]; ++k) {
             int clause_included_pos = clause_pos_base + k*2;

             //printf("\tI: %d (%d)\n", clause_bank_included[clause_included_pos], clause_bank_included[clause_included_pos + 1]);
        }

        for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
             int clause_excluded_pos = clause_pos_base + k*2;

             //printf("\tE: %d (%d)\n", clause_bank_excluded[clause_excluded_pos], clause_bank_excluded[clause_excluded_pos + 1]);
        }


        // Calculate intersection of input and included sets...
        // Go through the elements in the input set, one element at a time,
        // and calculate intersection of input set and included concept sets
        for (int k = 0; k < input_set_number_of_indices; ++k) {
            unsigned int element_chunk = input_set_indices[k] / 32;
            unsigned int element_pos = input_set_indices[k] % 32;
            
            // Check whether the input element is an element in all of the sets included by the clause (a match)
            unsigned int match = 1;
            for (int l = 0; l < clause_bank_included_length[j]; ++l) {
                unsigned int concept_set = clause_bank_included[clause_pos_base + l*2];

                if ((concept_sets[concept_set*number_of_element_chunks + element_chunk] & (1U << element_pos)) == 0) {
                    match = 0;
                    break;
                }
            }

            if (match) {
                set_intersection[element_chunk] |= (1U << element_pos);
            }
        }

        // Go through excluded concept sets and determine overlap with intersection of input and included concept sets
        // If there is an overlap, set the excluded concept set to True. Otherwise, set it to False

        for (int k = 0; k < number_of_concept_sets; ++k) {
            true_concept_sets[k] = 0;
        }

        for (int l = 0; l < clause_bank_excluded_length[j]; ++l) {
            unsigned int concept_set = clause_bank_excluded[clause_pos_base + l*2];

            for (int k = 0; k < input_set_number_of_indices; ++k) {
                unsigned int element = input_set_indices[k];
                unsigned int element_chunk = element / 32;
                unsigned int element_pos = element % 32;

                // At least one miss is needed to reinforce inclusion (reasoning by elimination)
                if (((concept_sets[concept_set*number_of_element_chunks + element_chunk] & (1U << element_pos)) == 0) &&
                    ((set_intersection[element_chunk] & (1U << element_pos)) > 0)) {
                    true_concept_sets[concept_set] = 1;
                    break;
                }
            }
        }

        // Remove elements from set intersection
        for (int k = 0; k < input_set_number_of_indices; ++k) {
            unsigned int element_chunk = input_set_indices[k] / 32;
            unsigned int element_pos = input_set_indices[k] % 32;

            set_intersection[element_chunk] &= ~(1U << element_pos);
        } 

        // Type II Feedback

		int k = clause_bank_excluded_length[j];
		while (k--) {
			int clause_excluded_pos = clause_pos_base + k*2;

            if (true_concept_sets[clause_bank_excluded[clause_excluded_pos]]) {
                clause_bank_excluded[clause_excluded_pos + 1] += 1;

                if (clause_bank_excluded[clause_excluded_pos + 1] >= number_of_states/2) {
                	int clause_included_pos = clause_pos_base + clause_bank_included_length[j]*2;
                    clause_bank_included[clause_included_pos] = clause_bank_excluded[clause_excluded_pos];
                    clause_bank_included[clause_included_pos + 1] = clause_bank_excluded[clause_excluded_pos + 1];
                    clause_bank_included_length[j] += 1;

                    clause_bank_excluded_length[j] -= 1;
                    int clause_excluded_end_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                    clause_bank_excluded[clause_excluded_pos] = clause_bank_excluded[clause_excluded_end_pos];
                    clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_excluded[clause_excluded_end_pos + 1];
                }
			}
  		}

        for (int k = 0; k < clause_bank_included_length[j]; ++k) {
             int clause_included_pos = clause_pos_base + k*2;

             //printf("\t>>I: %d (%d)\n", clause_bank_included[clause_included_pos], clause_bank_included[clause_included_pos + 1]);
        }

        for (int k = 0; k < clause_bank_excluded_length[j]; ++k) {
             int clause_excluded_pos = clause_pos_base + k*2;

             //printf("\t>>E: %d (%d)\n", clause_bank_excluded[clause_excluded_pos], clause_bank_excluded[clause_excluded_pos + 1]);
        }
    
        //fflush(stdout);
	}
}
