package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.stat.measure.Association;
import com.gengoai.apollo.stat.measure.ContingencyTable;
import com.gengoai.apollo.stat.measure.ContingencyTableCalculator;
import com.gengoai.collection.counter.HashMapMultiCounter;
import com.gengoai.collection.counter.MultiCounter;
import com.gengoai.stream.accumulator.MCounterAccumulator;
import com.gengoai.stream.accumulator.MMultiCounterAccumulator;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * <p>Uses a {@link ContingencyTableCalculator} to perform feature selection by taking the top N features per label
 * based on the calculator measure.</p>
 *
 * @author David B. Bracewell
 */
public class ContingencyFeatureSelection implements InstancePreprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private final int numFeaturesPerClass;
   private final double threshold;
   private final ContingencyTableCalculator calculator;

   /**
    * Instantiates a new Contingency feature selection.
    *
    * @param calculator          the calculator to use to generate statistics about features and labels
    * @param numFeaturesPerClass the num features per label
    * @param threshold           the minimum value from the calculator to accept
    */
   public ContingencyFeatureSelection(ContingencyTableCalculator calculator, int numFeaturesPerClass, double threshold) {
      this.calculator = calculator;
      this.numFeaturesPerClass = numFeaturesPerClass;
      this.threshold = threshold;
   }


   /**
    * A feature selector based on chi2 association metrics
    *
    * @param numFeaturesPerClass the num features per class
    * @param threshold           the threshold
    * @return the preprocessor
    */
   public static Preprocessor chi2Selector(int numFeaturesPerClass, double threshold) {
      return new ContingencyFeatureSelection(Association.CHI_SQUARE, numFeaturesPerClass, threshold);
   }

   /**
    * A feature selector based on g2 association metrics
    *
    * @param numFeaturesPerClass the num features per class
    * @param threshold           the threshold
    * @return the preprocessor
    */
   public static Preprocessor g2Selector(int numFeaturesPerClass, double threshold) {
      return new ContingencyFeatureSelection(Association.G_SQUARE, numFeaturesPerClass, threshold);
   }

   /**
    * A feature selector based on odds ratio association metrics
    *
    * @param numFeaturesPerClass the num features per class
    * @param threshold           the threshold
    * @return the preprocessor
    */
   public static Preprocessor oddsRatioSelector(int numFeaturesPerClass, double threshold) {
      return new ContingencyFeatureSelection(Association.ODDS_RATIO, numFeaturesPerClass, threshold);
   }


   @Override
   public Instance applyInstance(Instance example) {
      return example;
   }

   @Override
   public Dataset fitAndTransform(Dataset dataset) {
      final Set<String> finalFeatures = new HashSet<>();

      MCounterAccumulator<Object> labelCounts = dataset.getType().getStreamingContext().counterAccumulator();
      MMultiCounterAccumulator<String, Object> featureLabelCounts = dataset
                                                                       .getType()
                                                                       .getStreamingContext()
                                                                       .multiCounterAccumulator();
      dataset.stream().forEach(instance -> {
         Object label = instance.getLabel();
         labelCounts.add(label);
         MultiCounter<String, Object> localCounts = new HashMapMultiCounter<>();
         instance.getFeatureNameSpace().forEach(f -> localCounts.increment(f, label));
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

      return dataset.map(example -> example.mapInstance(instance -> {
         return instance.mapFeatures(f -> {
            if (finalFeatures.contains(f.name)) {
               return Optional.of(f);
            }
            return Optional.empty();
         });
      }));
   }

   @Override
   public void reset() {

   }

   @Override
   public String toString() {
      return "ContingencyFeatureSelection{measure=" + calculator + ", numFeaturesPerClass=" + numFeaturesPerClass + ", threshold=" + threshold + "}";
   }

}//END OF ContingencyFeatureSelection
