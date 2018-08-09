package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.collection.Maps;
import com.gengoai.conversion.Cast;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Collectors;

import static com.gengoai.tuple.Tuples.$;

/**
 * <p>Specialized list of {@link Preprocessor}s with added methods for reading, writing, and applying.</p>
 *
 * @param <T> the  example type parameter
 */
public final class PreprocessorList<T extends Example> extends ArrayList<Preprocessor<T>> implements JsonSerializable {
   private static final long serialVersionUID = 1L;


   /**
    * Instantiates a new empty Preprocessor list.
    */
   public PreprocessorList() {

   }

   /**
    * Instantiates a new preprocessor list using the given collection of preprocessors.
    *
    * @param preprocessors the preprocessors to add to the list
    */
   public PreprocessorList(@NonNull Collection<Preprocessor<T>> preprocessors) {
      super(preprocessors);
   }

   /**
    * Convenience method for creating an empty preprocessor list
    *
    * @param <T> the example type parameter
    * @return the empty preprocessor list
    */
   public static <T extends Example> PreprocessorList<T> empty() {
      return new PreprocessorList<>();
   }

   /**
    * Creates a new preprocessor list from the given preprocessor.
    *
    * @param <T>           the example type parameter
    * @param preprocessors the preprocessors
    * @return the preprocessor list
    */
   @SafeVarargs
   public static <T extends Example> PreprocessorList<T> create(Preprocessor<T>... preprocessors) {
      if (preprocessors == null) {
         return empty();
      } else {
         return new PreprocessorList<>(Arrays.asList(preprocessors));
      }
   }

   /**
    * Applies the preprocess in sequential order the given example
    *
    * @param example the example to preprocess
    * @return the preprocessed example
    */
   public T apply(T example) {
      if (isEmpty()) {
         return example;
      }
      T transformed = example;
      for (Preprocessor<T> preprocessor : this) {
         transformed = preprocessor.apply(transformed);
      }
      return transformed;
   }

   /**
    * Creates a new preprocessor list containing only the preprocessors that are needed when applying the model.
    *
    * @return the preprocessors required by the model
    */
   public PreprocessorList<T> getModelProcessors() {
      return new PreprocessorList<>(stream()
                                       .filter(p -> !p.trainOnly())
                                       .collect(Collectors.toList()));
   }

   /**
    * Resets all the preprocessors to an initial state.
    */
   public void reset() {
      forEach(Preprocessor::reset);
   }

   public static <T extends Example> PreprocessorList<T> fromJson(JsonEntry entry) {
      PreprocessorList<T> list = new PreprocessorList<>();
      entry.forEachElement(e -> {
         Class<?> clazz = e.getValProperty("class").asClass();
         list.add(Cast.as(e.getProperty("data", clazz)));
      });
      return list;
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry array = JsonEntry.array();
      for (Preprocessor<?> p : this) {
         array.addValue(Maps.hashMapOf($("class", p.getClass().getName()),
                                       $("data", p)
                                      ));
      }
      return array;
   }

}// END OF PreprocessorList
