package com.gengoai.apollo.hash;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.collection.multimap.HashSetMultimap;
import com.gengoai.collection.multimap.SetMultimap;
import com.gengoai.tuple.Tuple2;

import java.io.Serializable;
import java.util.Set;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class InMemoryLSHStorage implements LSHStorage, Serializable {
   private static final long serialVersionUID = 1L;
   private final SetMultimap<Tuple2<Integer, Integer>, NDArray> store = new HashSetMultimap<>();

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
