package com.gengoai.apollo.ml.encoder;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.stream.MStream;
import com.gengoai.stream.accumulator.MAccumulator;
import lombok.NonNull;

import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/**
 * <p>A label encoder that encodes each unique label object to an integer id.</p>
 *
 * @author David B. Bracewell
 */
public class LabelIndexEncoder extends IndexEncoder implements LabelEncoder {
   private static final long serialVersionUID = 1L;

   public LabelIndexEncoder() {

   }

   public LabelIndexEncoder(String... labels) {
      this.index.addAll(Arrays.asList(labels));
   }


   @Override
   public LabelEncoder createNew() {
      return new LabelIndexEncoder();
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
         MAccumulator<String, Set<String>> accumulator = dataset.getStreamingContext().setAccumulator();
         dataset.stream()
                .parallel()
                .flatMap(ex -> ex.getLabelSpace().map(Object::toString))
                .filter(Objects::nonNull)
                .forEach(accumulator::add);
         this.index.addAll(accumulator.value());
      }
   }
}// END OF LabelIndexEncoder
