package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Helper class for serializing and deserializing examples in JSON format Note: This could be made to use the Java
 * service provider framework to allow for extensions to example beyond Instance and Sequence.
 *
 * @author David B. Bracewell
 */
final class ExampleSerializer {

  /**
   * Writes an example to the given JSONWriter.
   *
   * @param writer  the writer
   * @param example the example
   * @throws IOException Something went wrong writing
   */
  static void write(@NonNull JSONWriter writer, @NonNull Example example) throws IOException {
    if (example instanceof Instance) {
      InstanceSerializer.write(writer, Cast.<Instance>as(example));
    } else if (example instanceof Sequence) {
      SequenceSerializer.write(writer, Cast.as(example));
    } else {
      throw new IllegalArgumentException(example.getClass().getName() + " is not supported.");
    }
  }

  /**
   * Reads an example from the given writer.
   *
   * @param reader the reader
   * @return the example
   * @throws IOException Something went wrong reading
   */
  static Example read(@NonNull JSONReader reader) throws IOException {
    if (reader.peek() == ElementType.BEGIN_OBJECT) {
      return InstanceSerializer.read(reader);
    } else if (reader.peek() == ElementType.BEGIN_ARRAY) {
      return SequenceSerializer.read(reader);
    }
    throw new IOException("Expecting BEGIN_OBJECT or BEGIN_ARRAY, but found " + reader.peek());
  }

  private interface SequenceSerializer {
    static void write(JSONWriter writer, Sequence sequence) throws IOException {
      writer.beginArray();
      for (Instance instance : sequence.asInstances()) {
        InstanceSerializer.write(writer, instance);
      }
      writer.endArray();
    }

    static Sequence read(JSONReader reader) throws IOException {
      ArrayList<Instance> instances = new ArrayList<>();
      reader.beginArray();
      while (reader.peek() != ElementType.END_ARRAY) {
        instances.add(InstanceSerializer.read(reader));
      }
      reader.endArray();
      instances.trimToSize();
      return new Sequence(instances);
    }

  }

  private interface InstanceSerializer {
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
