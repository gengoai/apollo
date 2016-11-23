package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MDoubleAccumulator;
import com.davidbracewell.string.StringUtils;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * <p>Transform values using <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf">Tf-idf</a> </p>
 *
 * @author David B. Bracewell
 */
public class TFIDFTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private volatile Counter<String> documentFrequencies = Counters.newCounter();
   private volatile double totalDocs = 0;

   /**
    * Instantiates a new Tfidf transform.
    */
   public TFIDFTransform() {
      super(StringUtils.EMPTY);
   }


   /**
    * Instantiates a new Tfidf transform.
    *
    * @param featureNamePrefix the feature name prefix
    */
   public TFIDFTransform(String featureNamePrefix) {
      super(featureNamePrefix);
   }


   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      double dSum = originalExample.getFeatures().stream().mapToDouble(Feature::getValue).sum();
      return featureStream.map(f -> {
                                  double value = f.getValue() / dSum * Math.log(totalDocs / documentFrequencies.get(f.getName()));
                                  if (value != 0) {
                                     return Feature.real(f.getName(), value);
                                  }
                                  return null;
                               }
                              )
                          .filter(Objects::nonNull);
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      MDoubleAccumulator docCount = stream.getContext().doubleAccumulator(0d);
      this.documentFrequencies.merge(stream.flatMap(instance -> {
                                                       docCount.add(1d);
                                                       return instance.stream().map(Feature::getName).distinct();
                                                    }
                                                   ).countByValue()
                                    );
      this.totalDocs = docCount.value();
   }

   @Override
   public void reset() {
      totalDocs = 0;
      documentFrequencies.clear();
   }

   @Override
   public String describe() {
      if (applyToAll()) {
         return "TFIDFTransform{totalDocuments=" + totalDocs + ", vocabSize=" + documentFrequencies.size() + "}";
      }
      return "TFIDFTransform[" + getRestriction() + "]{totalDocuments=" + totalDocs + ", vocabSize=" + documentFrequencies
                                                                                                          .size() + "}";
   }

   @Override
   public void write(@NonNull StructuredWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.writeKeyValue("restriction", getRestriction());
      }
      writer.writeKeyValue("totalDocuments", totalDocs);
      writer.beginObject("documentCounts");
      for (Map.Entry<String, Double> entry : documentFrequencies.entries()) {
         writer.writeKeyValue(entry.getKey(), entry.getValue());
      }
      writer.endObject();
   }

   @Override
   public void read(@NonNull StructuredReader reader) throws IOException {
      reset();
      while (reader.peek() != ElementType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "totalDocuments":
               this.totalDocs = reader.nextKeyValue().v2.asDoubleValue();
               break;
            case "documentCounts":
               reader.beginObject();
               while (reader.peek() != ElementType.END_OBJECT) {
                  Tuple2<String, Val> kv = reader.nextKeyValue();
                  documentFrequencies.set(kv.getKey(), kv.getValue().asDoubleValue());
               }
               reader.endObject();
               break;
         }
      }
   }
}// END OF TFIDFTransform
