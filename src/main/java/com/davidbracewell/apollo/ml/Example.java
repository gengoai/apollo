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

public interface Example extends Copyable<Example> {

  static Example read(@NonNull JSONReader reader) throws IOException {
    return ExampleSerializer.read(reader);
  }

  static Example fromString(String input) {
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

  default void write(@NonNull JSONWriter writer) throws IOException {
    ExampleSerializer.write(writer, this);
  }

  Stream<String> getFeatureSpace();

  Stream<Object> getLabelSpace();

  default String asString() {
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

  Example intern(Interner<String> interner);

}//END OF Example
