# -*- coding: utf-8 -*-

def modify_message(message, in_place, **to_modify_fields):
    if not in_place:
        new_message = message.__class__()
        new_message.CopyFrom(message)
    else:
        new_message = message
    for field, value in to_modify_fields.iteritems():
        _modify_message_per_field_in_place(new_message, field, value)
    return new_message


def _modify_message_per_field_in_place(message, field, value):
    field_list = field.split('.', 1)
    if len(field_list) > 1:
        _modify_message_per_field_in_place(getattr(message, field_list[0]),
                                           field_list[1], value)
    else:
        message.ClearField(field)
        if isinstance(value, list):
            getattr(message, field).extend(value)
        else:
            setattr(message, field, value)
