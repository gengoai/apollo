package com.gengoai.apollo.ml;

import com.gengoai.conversion.Cast;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The type Feature extractor.
 *
 * @param <I> the type parameter
 * @author David B. Bracewell
 */
public class FeatureExtractor<I> implements Serializable {
   private static final long serialVersionUID = 1L;
   private final List<Featurizer<I>> extractors;
   private final List<ContextFeature> contextFeatures;

   /**
    * Pipeline feature extractor.
    *
    * @param <I>        the type parameter
    * @param extractors the extractors
    * @return the feature extractor
    */
   public static <I> FeatureExtractor<I> pipeline(Object... extractors) {
      List<Featurizer<I>> featureExtractors = new ArrayList<>();
      List<ContextFeature> contextFeatures = new ArrayList<>();
      for (Object extractor : extractors) {
         if (extractor instanceof Featurizer) {
            featureExtractors.add(Cast.as(extractor));
         } else if (extractor instanceof ContextFeature) {
            contextFeatures.add(Cast.as(extractor));
         } else {
            throw new IllegalArgumentException();
         }
      }
      return new FeatureExtractor<>(featureExtractors, contextFeatures);
   }


   /**
    * Chain feature extractor.
    *
    * @param <I>        the type parameter
    * @param extractors the extractors
    * @return the feature extractor
    */
   @SafeVarargs
   public static <I> FeatureExtractor<I> chain(Featurizer<I>... extractors) {
      return new FeatureExtractor<>(Arrays.asList(extractors), Collections.emptyList());
   }

   /**
    * Instantiates a new Feature extractor.
    *
    * @param extractors      the extractors
    * @param contextFeatures the context features
    */
   protected FeatureExtractor(List<Featurizer<I>> extractors, List<ContextFeature> contextFeatures) {
      this.extractors = extractors;
      this.contextFeatures = contextFeatures;
   }


   /**
    * Extract instance example.
    *
    * @param datum the datum
    * @return the example
    */
   public final Example extractInstance(LabeledDatum<? extends I> datum) {
      Example ii = extractInstance(datum.data);
      ii.setLabel(datum.label);
      return ii;
   }

   /**
    * Extract instance example.
    *
    * @param input the input
    * @return the example
    */
   public final Example extractInstance(I input) {
      List<Feature> features = new ArrayList<>();
      for (Featurizer<I> extractor : extractors) {
         features.addAll(extractor.apply(input));
      }
      return new Instance(null, features);
   }

   /**
    * Extract sequence example.
    *
    * @param sequence the sequence
    * @return the example
    */
   public final Example extractSequence(LabeledSequence<? extends I> sequence) {
      Sequence out = new Sequence();
      sequence.forEach(i -> out.add(extractInstance(i)));
      return contextualize(out);
   }

   /**
    * Contextualize example.
    *
    * @param sequence the sequence
    * @return the example
    */
   public final Example contextualize(Example sequence) {
      for (ContextFeature contextFeature : contextFeatures) {
         sequence = contextFeature.apply(sequence);
      }
      return sequence;
   }

   /**
    * Extract sequence example.
    *
    * @param sequence the sequence
    * @return the example
    */
   public final Example extractSequence(List<? extends I> sequence) {
      Sequence out = new Sequence();
      for (I i : sequence) {
         out.add(extractInstance(i));
      }
      return contextualize(out);
   }

}//END OF FeatureExtractionPipeline
