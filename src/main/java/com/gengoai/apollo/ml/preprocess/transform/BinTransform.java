package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.Primitives;
import com.gengoai.Validation;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.json.JsonEntry;
import com.gengoai.math.EnhancedDoubleStatistics;
import com.gengoai.stream.MStream;
import com.gengoai.stream.accumulator.MStatisticsAccumulator;
import com.gengoai.string.StringUtils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * <p>Converts a real valued feature into a number of binary features by creating a <code>bin</code> number of new
 * binary features. Creates <code>bin</code> number of equal sized bins that the feature values can fall into.</p>
 *
 * @author David B. Bracewell
 */
public class BinTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private double[] bins;


   /**
    * Instantiates a new BinTransform with no restriction
    *
    * @param numberOfBins the number of bins to convert the feature into
    */
   public BinTransform(int numberOfBins) {
      Validation.checkArgument(numberOfBins > 0, "Number of bins must be > 0.");
      this.bins = new double[numberOfBins];
   }

   /**
    * Instantiates a new BinTransform.
    *
    * @param featureNamePrefix the feature name prefix to restrict to
    * @param numberOfBins      the number of bins to convert the feature into
    */
   public BinTransform(String featureNamePrefix, int numberOfBins) {
      super(featureNamePrefix);
      Validation.checkArgument(numberOfBins > 0, "Number of bins must be > 0.");
      this.bins = new double[numberOfBins];
   }

   /**
    * Instantiates a new Real to discrete transform.
    */
   protected BinTransform() {
      this(StringUtils.EMPTY, 1);
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      MStatisticsAccumulator stats = stream.getContext().statisticsAccumulator();
      stream.forEach(instance -> stats.combine(instance.stream()
                                                       .mapToDouble(Feature::getValue)
                                                       .collect(EnhancedDoubleStatistics::new,
                                                                EnhancedDoubleStatistics::accept,
                                                                EnhancedDoubleStatistics::combine)));
      EnhancedDoubleStatistics statistics = stats.value();
      double max = statistics.getMax();
      double min = statistics.getMin();
      double binSize = ((max - min) / bins.length);
      double sum = min;
      for (int i = 0; i < bins.length; i++) {
         sum += binSize;
         bins[i] = sum;
      }
   }

   @Override
   public void reset() {
      if (bins != null) {
         Arrays.fill(bins, 0);
      }
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.map(f -> {
         for (int i = 0; i < bins.length; i++) {
            if (f.getValue() < bins[i]) {
               return Feature.TRUE(f.getFeatureName(), Integer.toString(i));
            }
         }
         return Feature.TRUE(f.getFeatureName(), Integer.toString(bins.length - 1));
      });
   }


   @Override
   public String describe() {
      if (applyToAll()) {
         return "BinTransform{numberOfBins=" + bins.length + "}";
      }
      return "BinTransform[" + getRestriction() + "]{numberOfBins=" + bins.length + "}";
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry object = JsonEntry.object();
      if (!applyToAll()) {
         object.addProperty("restriction", getRestriction());
      }
      object.addProperty("bins", bins);
      return object;
   }

   public static BinTransform fromJson(JsonEntry entry) {
      BinTransform filter = new BinTransform();
      filter.setRestriction(entry.getStringProperty("restriction", null));
      filter.bins = Primitives.toDoubleArray(entry.getProperty("bins").asArray(Double.class));
      return filter;
   }


}// END OF RealToDiscreteTransform
