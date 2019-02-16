package com.gengoai.apollo.linear.store;

/**
 * @author David B. Bracewell
 */
public enum VectorStoreType {
   InMemory {
      @Override
      protected VSBuilder newBuilder(VectorStoreParameter parameters) {
         return new InMemoryVectorStore.Builder(parameters);
      }
   },
   IndexedFile {
      @Override
      protected VSBuilder newBuilder(VectorStoreParameter parameters) {
         return new DiskBasedVectorStore.Builder(parameters);
      }
   };


   protected abstract VSBuilder newBuilder(VectorStoreParameter parameters);

   public final VSBuilder builder(VectorStoreParameter parameters) {
      return newBuilder(parameters);
   }

}//END OF VectorStoreType
