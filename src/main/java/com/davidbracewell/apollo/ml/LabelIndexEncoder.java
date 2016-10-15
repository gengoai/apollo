package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MAccumulator;
import lombok.NonNull;

import java.util.Objects;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class LabelIndexEncoder extends IndexEncoder implements LabelEncoder {
   private static final long serialVersionUID = 1L;


   @Override
   public void fit(@NonNull Dataset<? extends Example> dataset) {
      if (!isFrozen()) {
         MAccumulator<String, Set<String>> accumulator = dataset.getStreamingContext().setAccumulator();
         dataset.stream()
                .parallel()
                .flatMap(ex -> ex.getLabelSpace().map(Object::toString))
                .filter(Objects::nonNull)
                .forEach(accumulator::add);
         this.index.addAll(accumulator.value());
      }
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
   public LabelEncoder createNew() {
      return new LabelIndexEncoder();
   }
}// END OF LabelIndexEncoder
