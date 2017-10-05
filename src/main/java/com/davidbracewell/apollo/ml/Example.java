package com.davidbracewell.apollo.ml;

import com.davidbracewell.Copyable;
import com.davidbracewell.Interner;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.json.JsonReader;
import com.davidbracewell.json.JsonSerializable;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;

/**
 * <p>Generic interface for representing a label and set of features. Classification and Regression problems use the
 * <code>Instance</code> specialization and Sequence Labeling Problems use the <code>Sequence</code>
 * specialization.</p>
 */
public interface Example extends Copyable<Example>, JsonSerializable {

   /**
    * Wraps reading an example from a JSON string as written by the write method. Useful for Datasets that keep examples
    * on disk or other places where serialization is required.
    *
    * @param input the input string
    * @return the example
    * @throws IOException Something went wrong converting the string into an example
    */
   static <T extends Example> T fromJson(String input, @NonNull Class<T> example) throws IOException {
      Preconditions.checkArgument(!StringUtils.isNullOrBlank(input),
                                  "Cannot create example from null or empty string.");
      Resource r = Resources.fromString(input);
      T rval;
      try (JsonReader reader = new JsonReader(r)) {
         reader.beginDocument();
         rval = reader.nextValue(example);
         reader.endDocument();
      }
      return rval;
   }

   /**
    * Returns the example as a list of instances
    *
    * @return The example as a list of instances
    */
   List<Instance> asInstances();

   /**
    * Gets the feature space of the example. The feature space is the set of distinct feature names in the example.
    *
    * @return the feature space
    */
   Stream<String> getFeatureSpace();

   /**
    * Gets the label space.
    *
    * @return the label space
    */
   Stream<Object> getLabelSpace();

   /**
    * Interns the feature space returning a new example whose feature names are interned.
    *
    * @param interner the interner
    * @return the example
    */
   Example intern(Interner<String> interner);

}//END OF Example
