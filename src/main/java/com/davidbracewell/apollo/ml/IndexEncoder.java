package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.collection.index.HashMapIndex;
import com.davidbracewell.collection.index.Index;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MAccumulator;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An encoder backed by an <code>Index</code> allowing a finite number of objects to be mapped to double values.
 *
 * @author David B. Bracewell
 */
public class IndexEncoder implements Encoder, Serializable {
   private static final long serialVersionUID = 1L;
   protected volatile Index<String> index = new HashMapIndex<>();
   protected volatile AtomicBoolean frozen = new AtomicBoolean(false);

   @Override
   public Encoder createNew() {
      return new IndexEncoder();
   }

   @Override
   public Object decode(double value) {
      return index.get((int) value);
   }

   @Override
   public double encode(Object object) {
      if (object == null) {
         return -1;
      }
      if (object instanceof Collection) {
         Collection<?> collection = Cast.as(object);
         double idx = -1;
         for (Object o : collection) {
            if (!frozen.get()) {
               idx = index.add(o.toString());
            } else {
               idx = index.getId(o.toString());
            }
         }
         return idx;
      }
      String str = object.toString();
      if (str != null) {
         if (!frozen.get()) {
            return index.add(str);
         }
      }
      return index.getId(str);
   }

   @Override
   public void fit(MStream<String> stream) {
      if (!isFrozen()) {
         MAccumulator<String, Set<String>> accumulator = stream.getContext().setAccumulator();
         stream.parallel()
               .filter(Objects::nonNull)
               .forEach(accumulator::add);
         this.index.addAll(accumulator.value());
      }
   }

   @Override
   public void fit(@NonNull Dataset<? extends Example> dataset) {
      if (!isFrozen()) {
//         MAccumulator<String, Set<String>> accumulator = dataset.getStreamingContext().setAccumulator();
//         dataset.stream()
//                .parallel()
//                .flatMap(ex -> ex.getFeatureSpace().map(Object::toString))
//                .filter(Objects::nonNull)
//                .forEach(accumulator::add);
//         this.index.addAll(accumulator.value());
      }
   }

   @Override
   public void freeze() {
      frozen.set(true);
   }

   @Override
   public double get(Object object) {
      if (object == null) {
         return -1;
      } else if (object instanceof Collection) {
         Collection<?> collection = Cast.as(object);
         double idx = -1;
         for (Object o : collection) {
            idx = index.getId(o.toString());
         }
         return idx;
      }
      return index.getId(object.toString());
   }

   @Override
   public boolean isFrozen() {
      return frozen.get();
   }

   @Override
   public int size() {
      return index.size();
   }

   @Override
   public void unFreeze() {
      frozen.set(false);
   }

   @Override
   public List<Object> values() {
      return Cast.cast(index.asList());
   }


}// END OF IndexEncoder
