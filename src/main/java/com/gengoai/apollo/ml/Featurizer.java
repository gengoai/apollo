package com.gengoai.apollo.ml;

import com.gengoai.collection.counter.Counter;
import com.gengoai.function.SerializableFunction;
import com.gengoai.function.SerializablePredicate;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>Defines methodology to extract {@link Feature}s from a given object.</p>
 *
 * @param <I> the type of the object extracting features from
 * @author David B. Bracewell
 */
public abstract class Featurizer<I> implements Serializable {
   private static final long serialVersionUID = 1L;


   /**
    * Creates a feature extractor that returns a single feature of the form <code>featurePrefix=function.apply(input)</code>.
    * If the function returns a null value no feature is generated.
    *
    * @param <I>           the type parameter
    * @param featurePrefix the feature prefix
    * @param function      the function
    * @return the featurizer
    */
   public static <I> Featurizer<I> valueFeaturizer(String featurePrefix, SerializableFunction<? super I, String> function) {
      return new ValueExtractor<>(featurePrefix, function);
   }

   /**
    * Creates a boolean feature extractor that uses the given function to convert an object into a collection of string
    * representing the feature names in form <code>PREFIX=NAME</code>.
    *
    * @param <I>      the type of the object extracting features from
    * @param function the function to convert the object into feature names
    * @return the featurizer
    */
   public static <I> Featurizer<I> booleanFeaturizer(SerializableFunction<? super I, ? extends Collection<String>> function) {
      return new BooleanExtractor<>(function);
   }

   /**
    * Creates a feature extractor that creates a true boolean feature with a given name when the given predicate
    * evaluates to true.
    *
    * @param <I>         the type of the object extracting features from
    * @param featureName the name of the feature to create (in the form of <code>PREFIX=NAME</code>)
    * @param predicate   the predicate to test the input object.
    * @return the featurizer
    */
   public static <I> Featurizer<I> predicateFeaturizer(String featureName, SerializablePredicate<? super I> predicate) {
      return new PredicateExtractor<>(featureName, predicate);
   }

   /**
    * Creates a real feature extractor that uses the given function to convert an object into a Counter of string
    * representing the feature names in form <code>PREFIX=NAME</code> with associated real-values.
    *
    * @param <I>      the type of the object extracting features from
    * @param function the function to convert the object into a counter of feature names and values.
    * @return the featurizer
    */
   public static <I> Featurizer<I> realFeaturizer(SerializableFunction<? super I, ? extends Counter<String>> function) {
      return new RealExtractor<>(function);
   }

   /**
    * Applies the featurizer to the given input producing a list of {@link Feature}
    *
    * @param input the input object to extract for features from
    * @return the list of extracted {@link Feature}
    */
   public abstract List<Feature> apply(I input);

   private static class BooleanExtractor<I> extends Featurizer<I> {
      private static final long serialVersionUID = 1L;
      private final SerializableFunction<? super I, ? extends Collection<String>> function;

      private BooleanExtractor(SerializableFunction<? super I, ? extends Collection<String>> function) {
         this.function = function;
      }

      @Override
      public List<Feature> apply(I input) {
         return function.apply(input)
                        .stream()
                        .map(Feature::booleanFeature)
                        .collect(Collectors.toList());
      }
   }

   private static class ValueExtractor<I> extends Featurizer<I> {
      private static final long serialVersionUID = 1L;
      private final SerializableFunction<? super I, String> function;
      private final String featurePrefix;

      private ValueExtractor(String featurePrefix, SerializableFunction<? super I, String> function) {
         this.function = function;
         this.featurePrefix = featurePrefix;
      }

      @Override
      public List<Feature> apply(I input) {
         String value = function.apply(input);
         if (value == null) {
            return Collections.emptyList();
         }
         return Collections.singletonList(Feature.booleanFeature(featurePrefix, value));
      }
   }

   private static class PredicateExtractor<I> extends Featurizer<I> {
      private static final long serialVersionUID = 1L;
      private final Feature feature;
      private final SerializablePredicate<? super I> predicate;

      private PredicateExtractor(String featureName, SerializablePredicate<? super I> predicate) {
         this.predicate = predicate;
         this.feature = Feature.booleanFeature(featureName);
      }

      @Override
      public List<Feature> apply(I input) {
         if (predicate.test(input)) {
            return Collections.singletonList(feature);
         }
         return Collections.emptyList();
      }
   }

   private static class RealExtractor<I> extends Featurizer<I> {
      private static final long serialVersionUID = 1L;
      private final SerializableFunction<? super I, ? extends Counter<String>> function;

      private RealExtractor(SerializableFunction<? super I, ? extends Counter<String>> function) {
         this.function = function;
      }

      @Override
      public List<Feature> apply(I input) {
         Counter<String> counter = function.apply(input);
         List<Feature> features = new ArrayList<>();
         counter.forEach((k, v) -> features.add(Feature.realFeature(k, v)));
         return features;
      }
   }

}//END OF Featurizer
