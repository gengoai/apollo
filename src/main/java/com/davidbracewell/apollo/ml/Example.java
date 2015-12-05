package com.davidbracewell.apollo.ml;

import com.davidbracewell.Copyable;
import com.davidbracewell.collection.Interner;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import com.davidbracewell.string.StringUtils;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import lombok.NonNull;

import java.io.IOException;
import java.util.stream.Stream;

/**
 * <p>Generic interface for representing a label and set of features. Classification and Regression problems use the
 * <code>Instance</code> specialization and Sequence Labeling Problems use the <code>Sequence</code>
 * specialization.</p>
 */
public interface Example extends Copyable<Example> {

  /**
   * Reads an example from a JSONReader. Only useful for reading examples written by a dataset.
   *
   * @param reader the reader
   * @return the example
   * @throws IOException Something went wrong reading.
   */
  static Example read(@NonNull JSONReader reader) throws IOException {
    return ExampleSerializer.read(reader);
  }

  /**
   * Wraps reading an example from a JSON string as written by the write method. Useful for Datasets that keep examples
   * on disk or other places where serialization is required.
   *
   * @param input the input string
   * @return the example
   * @throws IOException Something went wrong converting the string into an example
   */
  static Example fromJson(String input) throws IOException {
    Preconditions.checkArgument(!StringUtils.isNullOrBlank(input), "Cannot create example from null or empty string.");
    Resource r = Resources.fromString(input);
    Example rval = null;
    try (JSONReader reader = new JSONReader(r)) {
      reader.beginDocument();
      rval = read(reader);
      reader.endDocument();
    }
    return rval;
  }

  /**
   * Writes the example out to the given JSONWriter. Used by Datasets.
   *
   * @param writer the writer
   * @throws IOException Something went wrong writing
   */
  default void write(@NonNull JSONWriter writer) throws IOException {
    ExampleSerializer.write(writer, this);
  }

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
   * Converts the example to a String encoded in JSON
   *
   * @return the string
   */
  default String toJson() {
    Resource r = Resources.fromString();
    try (JSONWriter writer = new JSONWriter(r, true)) {
      writer.beginDocument();
      write(writer);
      writer.endDocument();
    } catch (Exception e) {
      throw Throwables.propagate(e);
    }
    try {
      return r.readToString().trim();
    } catch (IOException e) {
      throw Throwables.propagate(e);
    }
  }

  /**
   * Interns the feature space returning a new example whose feature names are interned.
   *
   * @param interner the interner
   * @return the example
   */
  Example intern(Interner<String> interner);

}//END OF Example
