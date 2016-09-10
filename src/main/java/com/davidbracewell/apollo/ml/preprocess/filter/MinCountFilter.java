package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class MinCountFilter extends RestrictedInstancePreprocessor implements FilterProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private long minCount;
   private volatile Set<String> selectedFeatures = Collections.emptySet();

   public MinCountFilter(@NonNull String featurePrefix, long minCount) {
      super(featurePrefix);
      this.minCount = minCount;
   }

   public MinCountFilter(long minCount) {
      this.minCount = minCount;
   }

   protected MinCountFilter() {
      this.minCount = 0;
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      selectedFeatures = Counters.newCounter(stream
                                                       .flatMap(l -> l.stream()
                                                                      .map(Feature::getName)
                                                                      .collect(Collectors.toList()))
                                                       .countByValue()
                                            )
                                 .filterByValue(v -> v >= minCount)
                                 .items();
   }

   @Override
   public void reset() {
      selectedFeatures.clear();
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.filter(f -> selectedFeatures.contains(f.getName()));
   }

   @Override
   public String describe() {
      if (acceptAll()) {
         return "MinCountFilter{minCount=" + minCount + "}";
      }
      return "CountFilter[" + getRestriction() + "]{minCount=" + minCount + "}";
   }

   @Override
   public void write(@NonNull StructuredWriter writer) throws IOException {
      if (!acceptAll()) {
         writer.writeKeyValue("restriction", getRestriction());
      }
      writer.writeKeyValue("minCount", minCount);
      writer.writeKeyValue("selected", selectedFeatures);
   }

   @Override
   public void read(@NonNull StructuredReader reader) throws IOException {
      reset();
      while (reader.peek() != ElementType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "minCount":
               this.minCount = reader.nextKeyValue().v2.asIntegerValue();
               break;
            case "selected":
               reader.nextCollection(HashSet::new).forEach(val -> selectedFeatures.add(val.asString()));
               break;
         }
      }
   }
}// END OF CountFilter
