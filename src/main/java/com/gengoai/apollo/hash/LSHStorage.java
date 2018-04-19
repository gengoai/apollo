package com.gengoai.apollo.hash;

import com.gengoai.apollo.linear.NDArray;

import java.util.Set;

/**
 * The interface Lsh storage.
 *
 * @author David B. Bracewell
 */
public interface LSHStorage {

   /**
    * Adds the given NDArray id to the LSH table.
    *
    * @param vector the vector
    * @param band   the band
    * @param bucket the bucket
    * @return the int
    */
   void add(NDArray vector, int band, int bucket);

   /**
    * Clear.
    */
   void clear();

   /**
    * Get int [ ].
    *
    * @param band   the band
    * @param bucket the bucket
    * @return the int [ ]
    */
   Set<NDArray> get(int band, int bucket);

}// END OF LSHStorage
