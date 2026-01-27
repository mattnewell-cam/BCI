from pylsl import resolve_streams


streams = resolve_streams()
print(f"Found {len(streams)} LSL streams:")
for s in streams:
    print(
        f"- name={s.name()} type={s.type()} "
        f"ch={s.channel_count()} fs={s.nominal_srate()}"
    )