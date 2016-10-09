package com.davidbracewell.apollo.linalg;

import lombok.NonNull;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Lsh vector store.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public class InMemoryLSHVectorStore<KEY> extends LSHVectorStore<KEY> {
   private static final long serialVersionUID = 1L;
   private final AtomicInteger vectorIDGenerator = new AtomicInteger();
   private final OpenIntObjectHashMap<LabeledVector> vectorIDMap = new OpenIntObjectHashMap<>();
   private final OpenObjectIntHashMap<KEY> keys = new OpenObjectIntHashMap<>();

   /**
    * Instantiates a new Lsh vector store.
    *
    * @param lsh the lsh
    */
   public InMemoryLSHVectorStore(@NonNull InMemoryLSH lsh) {
      super(lsh);
   }


   @Override
   public Set<KEY> keySet() {
      return new HashSet<>(keys.keys());
   }

   @Override
   public boolean containsKey(KEY key) {
      return keySet().contains(key);
   }

   @Override
   protected void removeVector(LabeledVector vector, int id) {
      vectorIDMap.removeKey(id);
      keys.removeKey(vector.getLabel());
   }

   @Override
   public Iterator<LabeledVector> iterator() {
      return Collections.unmodifiableCollection(vectorIDMap.values()).iterator();
   }

   @Override
   protected void registerVector(LabeledVector vector, int id) {
      keys.put(vector.getLabel(), id);
      vectorIDMap.put(id, vector);
   }

   @Override
   protected int nextUniqueID() {
      return vectorIDGenerator.getAndIncrement();
   }

   @Override
   protected int getID(KEY key) {
      return keys.get(key);
   }

   @Override
   protected LabeledVector getVectorByID(int id) {
      return vectorIDMap.get(id);
   }

   @Override
   public int size() {
      return vectorIDMap.size();
   }

}// END OF LSHVectorStore
