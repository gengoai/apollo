package com.gengoai.apollo.ml.preprocess.filter;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.InstancePreprocessor;
import com.gengoai.apollo.stat.measure.Association;
import com.gengoai.apollo.stat.measure.ContingencyTable;
import com.gengoai.apollo.stat.measure.ContingencyTableCalculator;
import com.gengoai.collection.counter.HashMapMultiCounter;
import com.gengoai.collection.counter.MultiCounter;
import com.gengoai.json.JsonReader;
import com.gengoai.json.JsonTokenType;
import com.gengoai.json.JsonWriter;
import com.gengoai.stream.accumulator.MCounterAccumulator;
import com.gengoai.stream.accumulator.MMultiCounterAccumulator;
import lombok.Getter;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * <p>Uses a {@link ContingencyTableCalculator} to perform feature selection by taking the top N features per label
 * based on the calculator measure.</p>
 *
 * @author David B. Bracewell
 */
public class ContingencyFeatureSelection implements FilterProcessor<Instance>, InstancePreprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private final Set<String> finalFeatures = new HashSet<>();
   private ContingencyTableCalculator calculator;
   @Getter
   private int numFeaturesPerClass;
   @Getter
   private double threshold;

   /**
    * Instantiates a new Contingency feature selection.
    *
    * @param calculator          the calculator to use to generate statistics about features and labels
    * @param numFeaturesPerClass the num features per label
    * @param threshold           the minimum value from the calculator to accept
    */
   public ContingencyFeatureSelection(@NonNull ContingencyTableCalculator calculator, int numFeaturesPerClass, double threshold) {
      this.calculator = calculator;
      this.numFeaturesPerClass = numFeaturesPerClass;
      this.threshold = threshold;
   }

   /**
    * Instantiates a new Contingency feature selection.
    */
   protected ContingencyFeatureSelection() {

   }

   @Override
   public Instance apply(Instance example) {
      example.getFeatures().removeIf(f -> !finalFeatures.contains(f.getFeatureName()));
      return example;
   }

   @Override
   public String describe() {
      return getClass().getSimpleName() + "{calculator=" + calculator
                                                              .getClass()
                                                              .getSimpleName() + ", numberOfFeaturesPerClass=" + numFeaturesPerClass + ", threshold=" + threshold + "}";
   }

   @Override
   public void fit(@NonNull Dataset<Instance> dataset) {
      MCounterAccumulator<Object> labelCounts = dataset.getType().getStreamingContext().counterAccumulator();
      MMultiCounterAccumulator<String, Object> featureLabelCounts = dataset
                                                                       .getType()
                                                                       .getStreamingContext()
                                                                       .multiCounterAccumulator();
      dataset.stream().forEach(instance -> {
         Object label = instance.getLabel();
         labelCounts.add(label);
         MultiCounter<String, Object> localCounts = new HashMapMultiCounter<>();
         instance.getFeatureSpace().forEach(f -> localCounts.increment(f, label));
         featureLabelCounts.merge(localCounts);
      });

      double totalCount = labelCounts.value().sum();
      for (Object label : labelCounts.value().items()) {
         double labelCount = labelCounts.value().get(label);
         Map<String, Double> featureScores = new HashMap<>();
         featureLabelCounts.value().firstKeys().forEach(feature -> {
                                                           double featureLabelCount = featureLabelCounts.value().get(feature, label);
                                                           double featureSum = featureLabelCounts.value().get(feature).sum();
                                                           if (featureLabelCount > 0) {
                                                              double score = calculator.calculate(
                                                                 ContingencyTable.create2X2(featureLabelCount, labelCount, featureSum, totalCount)
                                                                                                 );
                                                              featureScores.put(feature, score);
                                                           }
                                                        }
                                                       );

         List<Map.Entry<String, Double>> entryList = featureScores.entrySet()
                                                                  .stream()
                                                                  .sorted(
                                                                     Map.Entry.<String, Double>comparingByValue().reversed())
                                                                  .filter(e -> e.getValue() >= threshold)
                                                                  .collect(Collectors.toList());

         if (entryList.size() > 0) {
            entryList
               .subList(0, Math.min(numFeaturesPerClass, entryList.size()))
               .forEach(e -> finalFeatures.add(e.getKey()));
         }
      }
   }

   @Override
   public void fromJson(@NonNull JsonReader reader) throws IOException {
      reset();
      while (reader.peek() != JsonTokenType.END_OBJECT) {
         switch (reader.peekName()) {
            case "calculator":
               this.calculator = Association.valueOf(reader.nextKeyValue().v2.asString());
               break;
            case "threshold":
               this.threshold = reader.nextKeyValue().v2.asDoubleValue();
               break;
            case "numFeaturesPerClass":
               this.numFeaturesPerClass = reader.nextKeyValue().v2.asIntegerValue();
               break;
         }
      }
   }

   @Override
   public void reset() {
      finalFeatures.clear();
   }

   @Override
   public void toJson(@NonNull JsonWriter writer) throws IOException {
      writer.property("calculator", calculator.toString());
      writer.property("numFeaturesPerClass", numFeaturesPerClass);
      writer.property("threshold", threshold);
   }

   @Override
   public String toString() {
      return describe();
   }

}// END OF ContingencyFeatureSelection
