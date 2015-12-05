package com.davidbracewell.apollo.ml;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * The type Example serializer.
 *
 * @author David B. Bracewell
 */
final class ExampleSerializer {

  /**
   * Write.
   *
   * @param writer  the writer
   * @param example the example
   * @throws IOException the io exception
   */
  static void write(@NonNull JSONWriter writer, @NonNull Example example) throws IOException {
    if (example instanceof Instance) {
      InstanceSerializer.write(writer, Cast.<Instance>as(example));
    }
  }

  /**
   * Read example.
   *
   * @param reader the reader
   * @return the example
   * @throws IOException the io exception
   */
  static Example read(@NonNull JSONReader reader) throws IOException {
    if (reader.peek() == ElementType.BEGIN_OBJECT) {
      return InstanceSerializer.read(reader);
    }
    return null;
  }


  private interface InstanceSerializer {
    /**
     * Write.
     *
     * @param writer   the writer
     * @param instance the instance
     * @throws IOException the io exception
     */
    static void write(JSONWriter writer, Instance instance) throws IOException {
      writer.beginObject();
      writer.writeKeyValue("label", instance.getLabel());
      writer.beginObject("features");
      for (Feature f : instance) {
        writer.writeKeyValue(f.getName(), f.getValue());
      }
      writer.endObject();
      writer.endObject();
    }

    /**
     * Read instance.
     *
     * @param reader the reader
     * @return the instance
     * @throws IOException the io exception
     */
    static Instance read(JSONReader reader) throws IOException {
      Object label;
      List<Feature> features = new LinkedList<>();
      boolean openObject = reader.peek() == ElementType.BEGIN_OBJECT;
      if (openObject) {
        reader.beginObject();
      }
      label = reader.nextKeyValue("label").getValue().as(Object.class);
      reader.beginObject();
      while (reader.peek() != ElementType.END_OBJECT) {
        Tuple2<String, Val> keyValue = reader.nextKeyValue();
        features.add(Feature.real(keyValue.getKey(), keyValue.getValue().asDoubleValue()));
      }
      reader.endObject();
      if (openObject) {
        reader.endObject();
      }
      return new Instance(features, label);
    }
  }

}//END OF ExampleSerializer
