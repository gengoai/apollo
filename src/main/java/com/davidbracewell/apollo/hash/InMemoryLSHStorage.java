package com.davidbracewell.apollo.hash;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.guava.common.collect.HashMultimap;
import com.davidbracewell.tuple.Tuple2;

import java.io.Serializable;
import java.util.Set;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class InMemoryLSHStorage implements LSHStorage, Serializable {
   private static final long serialVersionUID = 1L;
   private final HashMultimap<Tuple2<Integer, Integer>, NDArray> store = HashMultimap.create();

   @Override
   public void add(NDArray vector, int band, int bucket) {
      store.put($(band, bucket), vector);
   }

   @Override
   public void clear() {
      store.clear();
   }

   @Override
   public Set<NDArray> get(int band, int bucket) {
      return store.get($(band, bucket));
   }

}// END OF InMemoryLSHStorage
