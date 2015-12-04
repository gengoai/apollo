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
 * The interface Example.
 */
public interface Example extends Copyable<Example> {

  /**
   * Read example.
   *
   * @param reader the reader
   * @return the example
   * @throws IOException the io exception
   */
  static Example read(@NonNull JSONReader reader) throws IOException {
    return ExampleSerializer.read(reader);
  }

  /**
   * From string example.
   *
   * @param input the input
   * @return the example
   */
  static Example fromJson(String input) {
    Preconditions.checkArgument(!StringUtils.isNullOrBlank(input), "Cannot create example from null or empty string.");
    Resource r = Resources.fromString(input);
    Example rval = null;
    try (JSONReader reader = new JSONReader(r)) {
      reader.beginDocument();
      rval = read(reader);
      reader.endDocument();
    } catch (Exception e) {
      throw Throwables.propagate(e);
    }
    return rval;
  }

  /**
   * Write.
   *
   * @param writer the writer
   * @throws IOException the io exception
   */
  default void write(@NonNull JSONWriter writer) throws IOException {
    ExampleSerializer.write(writer, this);
  }

  /**
   * Gets feature space.
   *
   * @return the feature space
   */
  Stream<String> getFeatureSpace();

  /**
   * Gets label space.
   *
   * @return the label space
   */
  Stream<Object> getLabelSpace();

  /**
   * As string string.
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
      return r.readToString().trim().replaceFirst("^\\[", "").replaceFirst("\\]$", "");
    } catch (IOException e) {
      throw Throwables.propagate(e);
    }
  }

  /**
   * Intern example.
   *
   * @param interner the interner
   * @return the example
   */
  Example intern(Interner<String> interner);

}//END OF Example
