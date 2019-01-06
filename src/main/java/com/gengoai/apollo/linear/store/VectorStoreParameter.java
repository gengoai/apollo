package com.gengoai.apollo.linear.store;

import com.gengoai.Parameters;
import com.gengoai.apollo.linear.hash.LSHParameter;
import com.gengoai.io.resource.Resource;

import java.io.Serializable;
import java.util.function.Consumer;

/**
 * @author David B. Bracewell
 */
public class VectorStoreParameter implements Parameters<VectorStoreParameter>, Serializable {
   public int cacheSize = 5000;
   public Resource location = null;
   public VectorStoreType type = VectorStoreType.InMemory;
   public LSHParameter lshParameters = null;


   public void lsh() {
      lshParameters = new LSHParameter();
   }

   public void lsh(Consumer<LSHParameter> updater) {
      lshParameters = new LSHParameter();
      updater.accept(lshParameters);
   }


}//END OF VectorStoreParameter
