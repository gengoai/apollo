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
    * Adds the given NDArray to the LSH table at the given band and bucket.
    *
    * @param vector the vector
    * @param band   the band
    * @param bucket the bucket
    */
   void add(NDArray vector, int band, int bucket);

   /**
    * Clears the storage.
    */
   void clear();

   /**
    * Gets the set of NDArrays at the given band and bucket
    *
    * @param band   the band
    * @param bucket the bucket
    * @return the set of NDArray associated with the given band and bucket
    */
   Set<NDArray> get(int band, int bucket);

}// END OF LSHStorage
