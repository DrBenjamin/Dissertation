-- Setting: Pandoc Lua filter to set `date` metadata to now + 4 hours (CET/CEST offset +02:00)

function Meta(meta)
  -- If the date metadata is not set or is the keyword 'now', override it
  local should_set = false
  if meta.date == nil then
    should_set = true
  else
    -- meta.date can be a MetaInlines/MetaString; convert to string safely
    local d = pandoc.utils.stringify(meta.date)
    if d == "now" then
      should_set = true
    end
  end

  if should_set then
    -- compute current time + 2 hours (CET offset)
    local t = os.time() + 2 * 60 * 60
    local iso = os.date("!%d.%m.%Y %H:%M", t)

    -- Try to get the last git commit message
    local commit_msg = nil
    local handle = io.popen("git log -1 --format=%s 2>/dev/null")
    if handle then
      commit_msg = handle:read("*a")
      handle:close()
      if commit_msg then
        commit_msg = commit_msg:gsub("%s+$", "") -- trim trailing whitespace
      end
      if commit_msg == "" then commit_msg = nil end
    end

    -- Build date with optional commit message below
    if commit_msg then
      meta.date = pandoc.MetaInlines({
        pandoc.Str(iso),
        pandoc.LineBreak(),
        pandoc.Emph({pandoc.Str("Last commit on https://github.com/DrBenjamin/Dissertation: " .. commit_msg)})
      })
    else
      meta.date = pandoc.MetaString(iso)
    end
  end

  return meta
end
