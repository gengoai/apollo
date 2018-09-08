package com.gengoai.apollo.linear.store.io;

import com.gengoai.apollo.linear.NDArray;

import java.io.IOException;

/**
 * @author David B. Bracewell
 */
public interface VectorStoreWriter extends AutoCloseable {


   int dimension();

   default void write(NDArray vector) throws IOException {
      write(vector.getLabel(), vector);
   }

   void write(String key, NDArray vector) throws IOException;


}//END OF VectorStoreWriter
